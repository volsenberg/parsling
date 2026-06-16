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
- Batch conversion via `parse_folder()`
- CLI: `parsling convert`, `parsling batch`, `parsling info`

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
parser = PdfParser(profile="vlm")
doc = parser.parse("complex_layout.pdf")
print(DocExporter(doc).to_markdown())
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
│   ├── exporters.py          # DocExporter — all output formats + rich MD builder
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
