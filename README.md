# parsling

A robust PDF parser built on **[Docling](https://docling-project.github.io/docling/)** for complex financial and government report layouts.

Designed for multi-layout PDFs — Indonesian government annual reports, financial statements, mixed-language documents with dense tables, charts, and figures.

---

## Features

- **3 preset profiles**: `FAST`, `ACCURATE`, `VLM`
- **TableFormer** (ACCURATE/FAST mode) for complex table extraction
- **Page-split table merging** — tables broken across pages are automatically detected and stitched into a single whole table
- **DocumentFigureClassifier-v2.5** — classifies figures as Chart, Diagram, Natural Image, etc.
- **CodeFormulaV2** — extracts code blocks and LaTeX formulas
- **EasyOCR** with Indonesian (`id`) language support (ACCURATE profile)
- **Tesseract** lightweight CPU OCR (FAST profile)
- **Granite-Docling-258M** end-to-end VLM pipeline (VLM profile)
- **Live per-page progress bar** — shows current stage (Rendering → OCR → Layout → Tables → Assembling) during parsing
- Export to Markdown, Rich Markdown, JSON, HTML/DocTags, CSV (per table), PNG figures
- **Rich Markdown recovery heuristics** — repairs known Docling extraction artifacts (dropped list items, page-split sentence fragments, headings fused into body text, unstable heading nesting) so output stays readable across very different document types (financial reports, dense legal/regulatory text) without per-document tuning. See [Rich Markdown Format](#rich-markdown-format).
- **Two-tier verification (`parsling verify`)** — free metadata-based triage to flag pages likely to have extraction artifacts, with an optional vision-LLM pass on just the flagged pages to catch what heuristics can't. See [Verification](#verification).
- Batch conversion via `parse_folder()`
- CLI: `parsling convert`, `parsling batch`, `parsling info`, `parsling verify`

---

## Profiles

| Profile | Pipeline | OCR | Table Mode | Enrichments |
|---|---|---|---|---|
| `fast` | StandardPdf | Tesseract | FAST | None |
| `accurate` | StandardPdf | EasyOCR `[en, id]` | ACCURATE | Figures + Formula + Code + Charts |
| `vlm` | VlmPipeline | Granite-Docling-258M | built-in | built-in |

> **`accurate`** is the recommended profile for financial reports.

---

## Installation

```bash
# 1. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install all dependencies
pip install -r requirements.txt

# 3. Install the parsling package itself (registers the CLI)
pip install -e .
```

Install Tesseract (required for the `fast` profile):

```bash
sudo apt install tesseract-ocr tesseract-ocr-ind
```

Install EasyOCR (required for the `accurate` profile — large download, ~1 GB):

```bash
pip install easyocr
```

Optional — for `parsling verify`'s vision-LLM tier, add an API key to a `.env` file in the project root (loaded automatically):

```bash
OPENAI_API_KEY=sk-...
```

---

## Quick Start

### Python API

```python
from pathlib import Path
from parsling import PdfParser, DocExporter

# Parse with default ACCURATE profile
parser = PdfParser()
doc = parser.parse("report.pdf")

# Export to standard Markdown + JSON + CSV tables
exp = DocExporter(doc)
exp.save(Path("./output/"), formats=["md", "json", "csv"])

# Export to Rich Markdown (section-aware, table captions, page metadata)
exp.save(Path("./output/"), formats=["rich_md"])

# Access tables as DataFrames
for i, df in enumerate(exp.to_tables()):
    print(f"Table {i+1}:\n", df.head())
```

### Rich Markdown directly

```python
exp = DocExporter(doc)
md = exp.to_rich_markdown()          # merge split tables, include all metadata
print(md)
```

### VLM Profile (Granite-Docling-258M)

```python
parser = PdfParser(profile="vlm", device="cuda")  # CPU by default — pass device="cuda" for GPU
doc = parser.parse("complex_layout.pdf")
print(DocExporter(doc).to_markdown())
```

```bash
parsling convert complex_layout.pdf --profile vlm --device cuda
```

> **GPU note:** `--device` defaults to `cpu` for every profile, VLM included — pass `--device cuda` explicitly. In testing this was ~3.5x faster (12s/page vs ~42s/page on CPU for Granite-Docling-258M).

> **VLM accuracy caveat:** unlike the OCR pipelines, the VLM is a small *generative* model — it occasionally hallucinates individual words during decoding (confirmed against ground truth: e.g. "dua puluh persen" → "dua puhul persen", "tunai" → "tuname"). This is silent corruption, not a visibly broken sentence, so it's a real risk for legal/financial documents where exact wording matters. The `accurate`/`fast` OCR profiles don't have this failure mode (their errors tend to be structural — page splits, fused blocks — which the rich-markdown recovery heuristics can detect and patch; word-level hallucination has no structural signature to catch).

You can override the VLM's default instruction prompt via `vlm_prompt=` (Python) or `--vlm-prompt` (CLI) — e.g. to add domain or language context. In testing, adding an Indonesian-language hint did **not** fix the hallucination above (same garbled word reproduced on the identical page with or without the hint), so treat this as a low-confidence lever rather than an accuracy fix:

```bash
parsling convert complex_layout.pdf --profile vlm --device cuda \
    --vlm-prompt "Convert this page to docling. The document is in Indonesian."
```

### Batch Conversion

```python
from parsling import PdfParser, DocExporter

parser = PdfParser(profile="accurate")
for result in parser.parse_folder(Path("./pdfs/")):
    if result.ok:
        DocExporter(result.document).save(
            output_dir=Path("./output/") / result.source.stem,
            formats=["json"],
        )
```

### Custom Profile

```python
from parsling import PdfParser
from parsling.config import ParseProfile

custom = ParseProfile(
    use_vlm=False,
    ocr_engine="easyocr",
    ocr_langs=["en", "id", "ms"],
    table_mode="accurate",
    do_picture_classification=True,
    do_chart_extraction=False,
    do_formula_enrichment=False,
    do_code_enrichment=False,
    accelerator="cuda",        # use GPU
    num_threads=8,
)
parser = PdfParser(profile=custom)
```

---

## CLI

The parser separates parsing (heavy, slow) from formatting (light, instant). PDFs are always parsed to a lossless JSON `DoclingDocument` first; all other formats are derived from that JSON.

### Step 1 — Parse PDF to JSON

```bash
# Default: accurate profile with live per-page progress bar
# Filenames are resolved from ./input/ automatically — no need to type the folder
parsling convert report.pdf

# Fast profile (Tesseract, no enrichments)
parsling convert report.pdf --profile fast

# Parse only a page range
parsling convert report.pdf --pages 1-10 --profile fast

# Write a log file alongside terminal output
parsling convert report.pdf --log output/report.log

# Batch parse everything in input/ (default folder)
parsling batch

# Batch parse a specific folder
parsling batch ./input/ --profile accurate

# Quick structure summary (no output saved)
parsling info report.pdf
```

During parsing you will see a live progress bar:

```
parsling · profile=fast · report.pdf

  171 pages detected

⠙ Page 42/171 · OCR  ████████████░░░░░░░░░  210/855  25%  0:04:18
```

### Step 2 — Export JSON to other formats

```bash
# Standard Markdown
parsling convert output/report_fast/report_fast.json --to md

# Rich Markdown (metadata-annotated, section breadcrumbs, merged split tables)
parsling convert output/report_fast/report_fast.json --to rich_md

# Multiple formats at once
parsling convert output/report_fast/report_fast.json --to md,rich_md,csv
```

#### Rich Markdown advanced flags

These flags only take effect when `rich_md` is included in `--to`:

```bash
# Keep page-split tables as separate fragments instead of merging them.
# Use this if you suspect a merge was incorrect and want to inspect the raw fragments.
parsling convert report.json --to rich_md --no-merge

# Omit <!-- FIGURE --> placeholder comments.
# Useful when you only care about text and tables and want a smaller file.
parsling convert report.json --to rich_md --no-pictures

# Widen the caption search window (default: 4 body positions).
# Raise this if captions in your PDF sit far away from their table.
parsling convert report.json --to rich_md --caption-window 8

# Raise the minimum text length filter (default: 5 characters).
# Increase if single-word OCR noise still appears; decrease if real short
# text like column labels ("No", "Rp") is being dropped.
parsling convert report.json --to rich_md --min-text 10
```

### Step 3 (optional) — Verify extraction quality

```bash
# Free heuristic triage, fully offline — flags suspect pages, no network call, no API key
parsling verify output/report_fast/report_fast.json

# Opt in to a vision-LLM pass on the flagged pages (needs network + the original PDF + an API key)
parsling verify output/report_fast/report_fast.json --llm --pdf report.pdf
```

The free triage is the default and requires nothing beyond the JSON you already have — parsling stays fully offline-capable. The vision-LLM tier is an opt-in add-on for when you have network access and want a second opinion on the pages the triage already narrowed down. See [Verification](#verification) below for details on what each tier checks and the cost/accuracy tradeoffs.

---

## Rich Markdown Format

`to_rich_markdown()` / `--to rich_md` produces Markdown enriched with HTML comments carrying structured metadata. The comments are invisible when rendered but fully parseable for downstream tools.

### YAML frontmatter

```markdown
---
document: report.pdf
schema: DoclingDocument v1.10.0
pages: 171
exported: 2026-06-16
---
```

### Section headings with page + depth

```markdown
## 3.1 Pelaksanaan APBN Tingkat Provinsi
<!-- section page=70 depth=2 -->
```

### Tables with section breadcrumb, caption, and page span

```markdown
<!-- TABLE 8: page=67-68, rows=21, cols=5, section="BAB II > 2.3 Reviu Capaian Kinerja", caption="Tabel 2.2 Hasil Reviu Efektivitas Kebijakan", merged_from_pages=67-68 -->
**Tabel 2.2 Hasil Reviu Efektivitas Kebijakan Makroekonomi dan Kesejahteraan Provinsi Aceh 2025**

| No. | Sasaran Makro Kesra | Target 2025 | Realisasi 2025 | Hasil Reviu |
|-----|---------------------|-------------|----------------|-------------|
| 1   | Pertumbuhan Ekonomi | 3,85%       | -1,61%         | ...         |
```

### Page-split table merging

Tables that Docling splits across page boundaries are automatically detected and stitched together:

- Detection: consecutive tables on adjacent pages with the same column count
- Repeated header rows on continuation pages are dropped
- `merged_from_pages=28-29` (or `90-91-92` for 3-page chains) is recorded in the comment
- Pass `--no-merge` to keep original fragments

### Body paragraphs

```markdown
Paragraph text here.
<!-- text page=70 section="BAB III > 3.1 Pelaksanaan APBN" -->
```

### Extraction-artifact recovery heuristics

Docling's raw `body.children` tree doesn't always reflect true reading order or even keep related content together — the rich-markdown builder walks the actual document structure (not just a flat list) and applies a few generic recovery passes so different document types (dense financial reports, heavily list-structured legal/regulatory text) come out readable without per-document tuning:

- **List-group flattening** — enumerated/lettered items (`a.`, `b.`, `(1)`, `(2)`, ...) live nested one level down inside `list` group containers, not directly under `body.children`. A flat walk silently drops everything inside them; the builder recurses into groups instead, preserving nesting depth and original markers.
- **Heading depth by "kind"** — a naive depth heuristic pushes every repeated heading pattern (`Pasal 1`, `Pasal 2`, ... `Pasal 117`) one level deeper each time, since each new heading falls back to "current stack depth + 1". The builder tracks depth per heading *kind* (its leading word — `Pasal`, `BAB`, `Bagian`, etc.) so a sequence of siblings stays flat.
- **Page-split continuation merging** — a page break can split one sentence or list clause into two separate body items. An unmarked list item or stray `text` item that looks like a sentence fragment (starts lowercase, no preceding section break) gets reattached to the paragraph it continues instead of rendering as a disconnected block.
- **Embedded-heading recovery** — sometimes a short heading-like line (e.g. a caption sitting directly above a paragraph) gets fused with unrelated text from a different line into a single block, because their bounding boxes were clustered together by the layout model. The builder detects an ALL-CAPS heading run fused as a prefix onto otherwise unrelated text, splits it out as its own heading, and reattaches the remainder to the paragraph it actually continues — using each item's bbox position to insert the heading *before* or *after* that paragraph, matching its true position in the source PDF rather than wherever Docling happened to order it.
- **Blank-line paragraph separation** — CommonMark joins consecutive non-blank lines into a single paragraph (soft line breaks collapse to spaces in most renderers). Every clause/list item is separated by a blank line so each one actually renders as its own paragraph or list item instead of collapsing into a wall of text.

None of these are document-type-specific — they're driven by structural signals (group nesting, bbox position, marker presence, text case) that hold regardless of whether the source is a financial report or a legal/regulatory document.

---

## Verification

Even with the recovery heuristics above, some extraction artifacts can't be confidently auto-fixed (Docling fusing two genuinely unrelated text regions with no recoverable boundary, OCR typos, truncated sentences). `parsling verify` is a two-tier check against the original PDF. **parsling is fully offline-capable** — the LLM tier is an optional add-on, never required.

**Tier 1 — free heuristic triage** (`flag_pages()`, always runs, fully offline): scans the JSON metadata for known artifact signatures — no network call, no API key, no cost. Flags pages with:
- an embedded heading still detected fused into body text
- unmarked list items (possible page-split continuation fragments)
- text that shrinks >30% after de-duplication (OCR repeat/garble)
- tables with no grid data

**Tier 2 — vision-LLM verification** (opt in via `--llm`, requires network + an API key, costs money): for only the pages Tier 1 flagged, renders that page from the original PDF and asks a vision-capable LLM to check the extracted text against the actual page image, reporting concrete discrepancies.

```bash
# Default: free triage only — no network, no API key, no cost
parsling verify output/report_accurate/report_accurate.json

# Opt in to the vision-LLM pass on flagged pages
parsling verify output/report_accurate/report_accurate.json --llm --pdf input/report.pdf

# Cap how many flagged pages get sent to the model (cost control)
parsling verify output/report_accurate/report_accurate.json --llm --pdf input/report.pdf --max-pages 5

# Save a JSON report
parsling verify output/report_accurate/report_accurate.json --llm --pdf input/report.pdf -o verify_report.json

# Use a different OpenAI-compatible vision model/provider
parsling verify output/report_accurate/report_accurate.json --llm --pdf input/report.pdf \
    --model gpt-4o --base-url https://api.openai.com/v1
```

The `--llm` tier requires an API key — defaults to `$OPENAI_API_KEY`, falling back to `$DEEPSEEK_API_KEY` (a `.env` file in the project root is loaded automatically). Default model is `gpt-5.4-nano` via OpenAI's API — chosen over `gpt-4o-mini` after head-to-head calibration on this project's documents showed fewer fabricated "issues" and more specific, accurate findings at a comparable cost.

> **Note on provider choice:** DeepSeek's API does **not** currently accept image input on its chat completions endpoint — confirmed directly against the live API (every model name tested returns the same `unknown variant image_url` deserialization error), despite some third-party claims otherwise. Use an OpenAI-compatible provider that actually supports vision (OpenAI itself is the default and is verified working).

> **Calibration caveat:** budget vision models are useful as a triage assistant, not an oracle. Even the better-calibrated `gpt-5.4-nano` still occasionally hallucinates an "issue" with no real discrepancy. `verify_document()` filters out the most obvious case (an issue whose `extracted_says` and `page_actually_says` are identical), but treat every reported issue as a candidate for manual review, not a confirmed bug.

---

## Output Structure

### After Step 1 (PDF → JSON)

```
output/
└── report_fast/
    ├── report_fast.json        ← lossless DoclingDocument (native format)
    └── images/
        ├── report_fast_chart_01.png
        └── …
```

### After Step 2 (JSON → formats)

```
output/
└── report_fast/
    ├── report_fast.json
    ├── report_fast.md          ← standard Markdown  (--to md)
    ├── report_fast_rich.md     ← metadata-rich Markdown  (--to rich_md)
    ├── report_fast_html.md     ← semantic HTML in fenced block  (--to html)
    ├── report_fast_doctags.md  ← DocTags XML in fenced block  (--to doctags)
    ├── tables/                 ← one CSV per table  (--to csv)
    │   ├── report_fast_table_01.csv
    │   └── …
    └── images/
        └── …
```

---

## Project Structure

```
parsling/
├── input/                    ← drop your PDFs here (git-ignored)
├── output/                   ← all parsed output lands here (git-ignored)
├── parsling/
│   ├── __init__.py           # Public API
│   ├── config.py             # ParseProfile dataclass
│   ├── profiles.py           # FAST, ACCURATE, VLM presets
│   ├── converter.py          # PdfParser — core parsing engine
│   ├── exporters.py          # DocExporter — all output formats + rich MD builder + recovery heuristics
│   ├── verify.py             # Two-tier verification: free heuristic triage + vision-LLM check
│   └── cli.py                # Typer CLI with live progress bar
├── examples/
│   ├── basic_usage.py
│   ├── financial_report.py
│   ├── batch_convert.py
│   └── json_to_rich_md.py   # Standalone JSON → Rich Markdown converter
├── tests/
│   ├── test_converter.py
│   └── test_exporters.py
├── requirements.txt
└── pyproject.toml
```

---

## Running Tests

```bash
pytest tests/ -v
```

`tests/test_rich_markdown.py` is a regression suite for the rich-markdown recovery heuristics, built from real extraction bugs found in production documents (a heading fused into a formula and rendered in the wrong order, heading-depth nesting growing unbounded, a person's name incorrectly fused with a trailing fragment, list items dropped because they're nested inside Docling group containers). Each fixture is a small synthetic `DoclingDocument`-shaped JSON, not a full PDF, so the suite runs in a couple seconds.

Runs automatically on every push/PR via [`.github/workflows/tests.yml`](.github/workflows/tests.yml) (GitHub Actions, free tier — no other CI setup needed).
