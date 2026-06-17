"""
parsling/cli.py
~~~~~~~~~~~~~~~
Typer-based command-line interface for parsling.

Commands
--------
    parsling convert  PATH   — Convert a single PDF
    parsling batch    FOLDER — Convert all PDFs in a folder
    parsling info     PATH   — Print document structure summary
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import queue
import re
import threading
from pathlib import Path

# Must be set before CUDA is initialized (i.e. before torch is imported by
# docling), so it has to happen at module import time, not inside a command.
# Reduces VRAM fragmentation that can turn a "should fit" allocation into an OOM.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from typing import Optional

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich import print as rprint

from dotenv import load_dotenv

load_dotenv()  # picks up HF_TOKEN (and friends) from a local .env file, if present

app = typer.Typer(
    name="parsling",
    help="PDF parser built on Docling — for complex financial and government report layouts.",
    add_completion=False,
)
console = Console()

_PROFILE_CHOICES = ["fast", "accurate", "vlm"]
_FORMAT_CHOICES = ["md", "rich_md", "json", "html", "csv", "doctags"]
_DEVICE_CHOICES = ["cpu", "cuda", "mps"]

from rich.logging import RichHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, show_path=False, console=console)],
)

# Suppress noisy library loggers by default
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("pdfminer").setLevel(logging.WARNING)
logging.getLogger("onnxruntime").setLevel(logging.ERROR)
logging.getLogger("docling.models.stages.ocr.tesseract_ocr_cli_model").setLevel(logging.CRITICAL)
# One INFO line per picture annotation migrated to the new `meta` format —
# pure noise, not actionable (docling_core internal schema migration).
logging.getLogger("docling_core.types.doc.document").setLevel(logging.WARNING)


def _count_pdf_pages(path: Path) -> int:
    """Return total page count without a full parse."""
    try:
        import pypdfium2
        doc = pypdfium2.PdfDocument(str(path))
        n = len(doc)
        doc.close()
        return n
    except Exception:
        return 0


# Stage order Docling emits — used to label the progress bar.
_STAGES = ["preprocess", "ocr", "layout", "table", "assemble"]
_STAGE_LABEL = {
    "preprocess": "Rendering",
    "ocr":        "OCR",
    "layout":     "Layout",
    "table":      "Tables",
    "assemble":   "Assembling",
}
# Regex matching: "PIPELINE_PROFILING Stage ocr: run_id=1 pages=[2, 3] ..."
_PROFILING_RE = re.compile(
    r"PIPELINE_PROFILING Stage (\w+): run_id=\d+ pages=\[([^\]]+)\]"
)


def _parse_with_progress(
    parser,
    path: Path,
    page_range: tuple[int, int] | None,
    total_pages: int,
) :
    """
    Run parser.parse() in a background thread and display a Rich progress bar
    driven by Docling's PIPELINE_PROFILING log messages.

    Returns the parsed DoclingDocument.
    """
    # Queue that the background thread's log handler pushes records into.
    log_queue: queue.Queue = queue.Queue()
    qh = logging.handlers.QueueHandler(log_queue)
    qh.setLevel(logging.DEBUG)

    pipeline_logger = logging.getLogger("docling.pipeline.standard_pdf_pipeline")
    pipeline_logger.addHandler(qh)
    pipeline_logger.setLevel(logging.DEBUG)
    # Stop pipeline DEBUG/INFO messages from bubbling up to the terminal —
    # the progress bar already conveys that information visually.
    pipeline_logger.propagate = False

    # Also silence noisy Docling sub-loggers during parsing.
    _quiet = [
        "docling.document_converter",
        "docling.pipeline.base_pipeline",
        "docling.models.factories",
        "docling.models.factories.base_factory",
        "docling.utils.accelerator_utils",
        "httpx",
        "httpcore",
    ]
    _saved_levels = {n: logging.getLogger(n).level for n in _quiet}
    _saved_propagate = {n: logging.getLogger(n).propagate for n in _quiet}
    for n in _quiet:
        logging.getLogger(n).setLevel(logging.WARNING)

    result: dict = {}

    def _run() -> None:
        try:
            result["doc"] = parser.parse(path, page_range=page_range)
        except Exception as exc:
            result["error"] = exc
        finally:
            pipeline_logger.removeHandler(qh)
            pipeline_logger.propagate = True
            for n in _quiet:
                logging.getLogger(n).setLevel(_saved_levels[n])

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    # Effective page count (may be a sub-range).
    if page_range:
        effective = page_range[1] - page_range[0] + 1
    else:
        effective = total_pages or 1

    # Each page goes through all 5 stages — total steps = pages × stages.
    total_steps = effective * len(_STAGES)
    completed_steps = 0
    current_stage = "Starting"
    latest_page = 0

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=36),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )

    with progress:
        task_id = progress.add_task(
            f"[cyan]Page 0/{effective} · {current_stage}",
            total=total_steps,
        )

        while thread.is_alive() or not log_queue.empty():
            try:
                record = log_queue.get(timeout=0.15)
                msg = record.getMessage()
                m = _PROFILING_RE.search(msg)
                if m:
                    stage = m.group(1)
                    pages = [p.strip() for p in m.group(2).split(",")]
                    current_stage = _STAGE_LABEL.get(stage, stage)
                    latest_page = max(latest_page, int(pages[-1]))
                    completed_steps += len(pages)
                    progress.update(
                        task_id,
                        completed=completed_steps,
                        description=(
                            f"[cyan]Page {latest_page}/{effective}"
                            f" · [yellow]{current_stage}"
                        ),
                    )
            except queue.Empty:
                pass

        progress.update(task_id, completed=total_steps,
                        description=f"[green]Done · {effective} pages")

    thread.join()

    if "error" in result:
        raise result["error"]
    return result["doc"]


def _add_file_logger(log_path: Path) -> None:
    """Add a plain-text FileHandler to the root logger."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logging.getLogger().addHandler(fh)
    logging.getLogger().info("Logging to file: %s", log_path)


def _parse_formats(formats_str: str) -> list[str]:
    """Parse comma-separated format string into a validated list."""
    parts = [f.strip().lower() for f in formats_str.split(",")]
    invalid = [f for f in parts if f not in _FORMAT_CHOICES]
    if invalid:
        rprint(f"[red]Unknown format(s): {invalid}. Choose from: {_FORMAT_CHOICES}[/red]")
        raise typer.Exit(1)
    return parts


def _get_unique_stem(output_dir: Path, base_stem: str) -> str:
    """Find a unique output folder name and stem by appending _1, _2, etc. if it exists."""
    stem = base_stem
    counter = 1
    while (output_dir / stem).exists():
        stem = f"{base_stem}_{counter}"
        counter += 1
    return stem


def _resolve_input(path: Path) -> Path:
    """
    Resolve an input path, falling back to ./input/<path> if the given
    path does not exist. Allows users to type just the filename:
      parsling convert report.pdf   →   ./input/report.pdf
    """
    if path.exists():
        return path
    candidate = Path("./input") / path
    if candidate.exists():
        return candidate
    return path  # return original so the "not found" error shows the user's input


# -----------------------------------------------------------------------
# convert
# -----------------------------------------------------------------------

@app.command()
def convert(
    path: Path = typer.Argument(..., help="Path to the PDF file."),
    profile: str = typer.Option(
        "accurate",
        "--profile", "-p",
        help=f"Parse profile: {_PROFILE_CHOICES}",
    ),
    output: Path = typer.Option(
        Path("./output"),
        "--output", "-o",
        help="Output directory.",
    ),
    device: str = typer.Option(
        "cpu",
        "--device", "-d",
        help=f"Accelerator device: {_DEVICE_CHOICES}. Use 'cuda' if you have an NVIDIA GPU — "
             "dramatically faster for OCR, TableFormer, and chart extraction.",
    ),
    batch_size: int = typer.Option(
        4,
        "--batch-size",
        min=1,
        help="Pages processed together per batch through OCR/layout/table models. "
             "Lower this (e.g. 1 or 2) on GPUs with limited VRAM to avoid CUDA OOM "
             "with --device cuda. Higher = more throughput, more peak memory.",
    ),
    no_formula_enrichment: bool = typer.Option(
        False,
        "--no-formula-enrichment",
        help="Disable CodeFormulaV2 formula/code enrichment. Frees GPU VRAM — "
             "useful on limited-VRAM GPUs if your document has no math/code blocks.",
    ),
    no_chart_extraction: bool = typer.Option(
        False,
        "--no-chart-extraction",
        help="Disable the Granite Vision chart-extraction model (~4-5GB VRAM, the "
             "single largest GPU consumer in the accurate profile). Charts are kept "
             "as cropped images but not converted to tabular data. Use this on "
             "limited-VRAM GPUs alongside --no-formula-enrichment.",
    ),
    vlm_prompt: Optional[str] = typer.Option(
        None,
        "--vlm-prompt",
        help="[vlm profile only] Override Granite-Docling's default instruction "
             "('Convert this page to docling.'). Note: in testing, adding a "
             "language hint did not fix the model's word-level hallucinations — "
             "treat this as a low-confidence lever, not an accuracy fix.",
    ),
    formats: str = typer.Option(
        "json",
        "--to",
        help="Comma-separated export formats: md, rich_md, json, html, csv, doctags. Default: json.",
    ),
    page_range: Optional[str] = typer.Option(
        None,
        "--pages",
        help="Page range to parse, e.g. '1-10' for the first 10 pages (1-indexed).",
    ),
    save_figures: bool = typer.Option(
        True,
        "--figures/--no-figures",
        help="Save extracted figure images as PNGs.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    log: Optional[Path] = typer.Option(
        None,
        "--log",
        help="Write a plain-text log file to this path. Omit to log to terminal only.",
    ),
    # --- Rich Markdown options (only used when rich_md is in --to) ---
    no_merge: bool = typer.Option(
        False,
        "--no-merge",
        help=(
            "[rich_md] Keep page-split tables as separate fragments. "
            "By default, consecutive tables on adjacent pages with the same column count "
            "are automatically stitched into one whole table."
        ),
    ),
    no_pictures: bool = typer.Option(
        False,
        "--no-pictures",
        help=(
            "[rich_md] Omit <!-- FIGURE --> placeholder comments from the output. "
            "Useful when you only care about text and tables."
        ),
    ),
    caption_window: int = typer.Option(
        4,
        "--caption-window",
        min=1,
        help=(
            "[rich_md] How many body positions before/after a table to search for "
            "a caption. Raise this if captions in your PDF sit far from their table. "
            "Default: 4."
        ),
    ),
    min_text: int = typer.Option(
        5,
        "--min-text",
        min=1,
        help=(
            "[rich_md] Minimum character length for a text fragment to be included. "
            "Filters short OCR noise (single symbols, stray punctuation) that appear "
            "on cover pages. Default: 5."
        ),
    ),
) -> None:
    """Convert a single PDF file to structured outputs."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        # Re-enable Tesseract OSD logs for debugging
        logging.getLogger(
            "docling.models.stages.ocr.tesseract_ocr_cli_model"
        ).setLevel(logging.DEBUG)

    if log:
        _add_file_logger(log)

    from parsling.converter import PdfParser
    from parsling.exporters import DocExporter

    path = _resolve_input(path)
    if not path.exists():
        rprint(f"[red]File not found: {path}[/red]")
        raise typer.Exit(1)

    is_json_input = path.suffix.lower() == ".json"
    if is_json_input and formats == "json":
        fmt_list = ["html", "doctags"]
    else:
        fmt_list = _parse_formats(formats)

    if not is_json_input and fmt_list != ["json"]:
        rprint("[red]Error: Direct conversion from PDF to formats other than JSON is disabled.[/red]")
        rprint("[yellow]Please parse the PDF to JSON first, then convert the JSON file to your desired formats.[/yellow]")
        rprint("[dim]Example:[/dim]")
        rprint(f"[dim]  parsling convert {path.name} --to json[/dim]")
        rprint(f"[dim]  parsling convert output/{path.stem}/{path.stem}.json --to {formats}[/dim]")
        raise typer.Exit(1)

    pr: tuple[int, int] | None = None
    if page_range:
        try:
            start, end = [int(x) for x in page_range.split("-")]
            pr = (start, end)
        except ValueError:
            rprint("[red]Invalid --pages format. Use 'start-end', e.g. '1-10'.[/red]")
            raise typer.Exit(1)

    profile = profile.lower()
    if profile not in _PROFILE_CHOICES:
        rprint(f"[red]Unknown profile {profile!r}. Choose from: {_PROFILE_CHOICES}[/red]")
        raise typer.Exit(1)

    device = device.lower()
    if device not in _DEVICE_CHOICES:
        rprint(f"[red]Unknown device {device!r}. Choose from: {_DEVICE_CHOICES}[/red]")
        raise typer.Exit(1)

    if path.suffix.lower() == ".json":
        console.print(f"\n[bold cyan]parsling[/bold cyan] · loading structured JSON · {path.name}\n")
        from docling_core.types.doc import DoclingDocument
        with console.status(f"[bold green]Loading {path.name}…"):
            doc = DoclingDocument.load_from_json(path)
    else:
        console.print(f"\n[bold cyan]parsling[/bold cyan] · profile=[yellow]{profile}[/yellow] · device=[yellow]{device}[/yellow] · {path.name}\n")
        total_pages = _count_pdf_pages(path)
        if total_pages:
            console.print(f"[dim]  {total_pages} pages detected[/dim]\n")
        parser = PdfParser(
            profile=profile,
            device=device,
            page_batch_size=batch_size,
            do_formula_enrichment=not no_formula_enrichment,
            do_code_enrichment=not no_formula_enrichment,
            do_chart_extraction=not no_chart_extraction,
            vlm_prompt=vlm_prompt,
        )
        doc = _parse_with_progress(parser, path, pr, total_pages)

    exporter = DocExporter(doc)
    if path.suffix.lower() == ".json":
        stem = path.stem
    else:
        base_stem = f"{path.stem}_{profile}"
        stem = _get_unique_stem(output, base_stem)

    out = exporter.save(
        output_dir=output / stem,
        stem=stem,
        formats=fmt_list,
        save_figures=save_figures,
        rich_md_kwargs=dict(
            merge_split_tables=not no_merge,
            include_pictures=not no_pictures,
            caption_window=caption_window,
            min_text_len=min_text,
        ),
    )

    # Summary table
    tbl = Table(title="Export Summary", show_header=True, header_style="bold magenta")
    tbl.add_column("Item", style="cyan")
    tbl.add_column("Value")
    tbl.add_row("Source", str(path))
    tbl.add_row("Profile", profile)
    tbl.add_row("Pages", str(len(list(doc.pages))))
    tbl.add_row("Tables found", str(len(doc.tables)))
    tbl.add_row("Figures found", str(len(list(doc.pictures))))
    tbl.add_row("Output dir", str(out))
    tbl.add_row("Formats", formats)
    console.print(tbl)


# -----------------------------------------------------------------------
# batch
# -----------------------------------------------------------------------

@app.command()
def batch(
    folder: Path = typer.Argument(Path("./input"), help="Folder containing PDF files. Default: ./input"),
    profile: str = typer.Option("accurate", "--profile", "-p"),
    device: str = typer.Option(
        "cpu",
        "--device", "-d",
        help=f"Accelerator device: {_DEVICE_CHOICES}.",
    ),
    batch_size: int = typer.Option(
        4,
        "--batch-size",
        min=1,
        help="Pages processed together per batch through OCR/layout/table models. "
             "Lower this (e.g. 1 or 2) on GPUs with limited VRAM to avoid CUDA OOM.",
    ),
    output: Path = typer.Option(Path("./output"), "--output", "-o"),
    formats: str = typer.Option("json", "--to"),
    glob: str = typer.Option("**/*.pdf", "--glob", help="Glob pattern for PDF discovery."),
    save_figures: bool = typer.Option(True, "--figures/--no-figures"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    log: Optional[Path] = typer.Option(
        None,
        "--log",
        help="Write a plain-text log file to this path.",
    ),
) -> None:
    """Convert all PDFs in a folder (recursive by default)."""
    if verbose:
        logging.getLogger().setLevel(logging.INFO)

    if log:
        _add_file_logger(log)

    from parsling.converter import PdfParser
    from parsling.exporters import DocExporter

    if not folder.is_dir():
        rprint(f"[red]Not a directory: {folder}[/red]")
        raise typer.Exit(1)

    fmt_list = _parse_formats(formats)
    if fmt_list != ["json"]:
        rprint("[red]Error: Direct batch conversion from PDF to formats other than JSON is disabled.[/red]")
        rprint("[yellow]Batch process only supports outputting JSON format. To get other formats, convert the individual JSON files.[/yellow]")
        raise typer.Exit(1)
    profile = profile.lower()
    device = device.lower()
    if device not in _DEVICE_CHOICES:
        rprint(f"[red]Unknown device {device!r}. Choose from: {_DEVICE_CHOICES}[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold cyan]parsling batch[/bold cyan] · profile=[yellow]{profile}[/yellow] · device=[yellow]{device}[/yellow] · {folder}\n")

    parser = PdfParser(profile=profile, device=device, page_batch_size=batch_size)
    success = fail = 0
    any_processed = False

    for result in parser.parse_folder(folder, glob=glob):
        any_processed = True
        if not result.ok:
            rprint(f"  [red]✗[/red] {result.source.name} — {result.error}")
            fail += 1
            continue
        try:
            exp = DocExporter(result.document)
            base_stem = f"{result.source.stem}_{profile}"
            stem = _get_unique_stem(output, base_stem)
            exp.save(
                output_dir=output / stem,
                stem=stem,
                formats=fmt_list,
                save_figures=save_figures,
            )
            rprint(f"  [green]✓[/green] {result.source.name}")
            success += 1
        except Exception as exc:
            rprint(f"  [red]✗[/red] {result.source.name} — export error: {exc}")
            fail += 1

    if not any_processed:
        rprint("[yellow]No PDFs found.[/yellow]")
        raise typer.Exit(0)

    console.print(
        f"\n[bold]Done:[/bold] {success} succeeded, {fail} failed — output → [cyan]{output}[/cyan]\n"
    )


# -----------------------------------------------------------------------
# info
# -----------------------------------------------------------------------

@app.command()
def info(
    path: Path = typer.Argument(..., help="Path to the PDF file."),
    profile: str = typer.Option("fast", "--profile", "-p", help="Profile for quick parse."),
) -> None:
    """Print a structure summary of a PDF without saving any output."""
    from parsling.converter import PdfParser

    path = _resolve_input(path)
    if not path.exists():
        rprint(f"[red]File not found: {path}[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold cyan]parsling info[/bold cyan] · {path.name}\n")

    with console.status("[bold green]Analysing…"):
        parser = PdfParser(profile=profile)
        doc = parser.parse(path)

    tbl = Table(title=f"Structure: {path.name}", show_header=True, header_style="bold blue")
    tbl.add_column("Property", style="cyan")
    tbl.add_column("Value")
    tbl.add_row("Pages", str(len(list(doc.pages))))
    tbl.add_row("Tables", str(len(doc.tables)))
    tbl.add_row("Figures / Pictures", str(len(list(doc.pictures))))
    console.print(tbl)

    if doc.tables:
        console.print("\n[bold]Table previews:[/bold]")
        from parsling.exporters import DocExporter
        exp = DocExporter(doc)
        for i, df in enumerate(exp.to_tables()):
            console.print(f"\n  [yellow]Table {i + 1}[/yellow] — {df.shape[0]} rows × {df.shape[1]} cols")
            console.print(df.head(3).to_string(index=False))
    console.print()


# -----------------------------------------------------------------------
# verify
# -----------------------------------------------------------------------

@app.command()
def verify(
    json_path: Path = typer.Argument(..., help="Path to the parsed DoclingDocument JSON."),
    pdf: Optional[Path] = typer.Option(
        None,
        "--pdf",
        help="Path to the original source PDF. Required when --llm is passed.",
    ),
    llm: bool = typer.Option(
        False,
        "--llm",
        help=(
            "Opt in to the vision-LLM pass on flagged pages (costs money, needs network "
            "+ an API key). Off by default — parsling works fully offline without it; "
            "the free heuristic triage runs either way."
        ),
    ),
    model: str = typer.Option(
        "gpt-5.4-nano",
        "--model",
        help="Vision-capable model name (OpenAI-compatible chat completions API). Only used with --llm.",
    ),
    base_url: str = typer.Option(
        "https://api.openai.com/v1",
        "--base-url",
        help="OpenAI-compatible API base URL.",
    ),
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        help="Defaults to $OPENAI_API_KEY, then $DEEPSEEK_API_KEY (.env is loaded automatically).",
    ),
    max_pages: Optional[int] = typer.Option(
        None,
        "--max-pages",
        min=1,
        help="Cap how many flagged pages are sent to the model — cost control.",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Save the verification report as JSON to this path.",
    ),
) -> None:
    """
    Flag suspect pages with a free metadata triage, optionally verify them with a vision LLM.

    Stage 1 (free, always runs, fully offline): scans the DoclingDocument
    JSON metadata for known extraction-artifact patterns (embedded
    headings, page-split fragments, empty table grids) — no network call,
    no API key needed.

    Stage 2 (paid, opt in via --llm): renders each flagged page from the
    original PDF and asks a vision-capable LLM to check the extracted text
    against the page image. parsling works fully offline without this —
    it's an optional add-on for when you want a second opinion on the
    pages the free triage already narrowed down.
    """
    from parsling.verify import flag_pages, verify_document

    json_path = _resolve_input(json_path)
    if not json_path.exists():
        rprint(f"[red]File not found: {json_path}[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold cyan]parsling verify[/bold cyan] · {json_path.name}\n")

    raw = json.loads(json_path.read_text(encoding="utf-8"))
    flags = flag_pages(raw)

    if not flags:
        rprint("[green]No suspect pages found by heuristic triage.[/green]")
        raise typer.Exit(0)

    tbl = Table(title="Heuristic Triage (free)", show_header=True, header_style="bold yellow")
    tbl.add_column("Page", style="cyan")
    tbl.add_column("Reasons")
    for page in sorted(flags):
        tbl.add_row(str(page), "\n".join(flags[page]))
    console.print(tbl)
    console.print(f"\n[dim]{len(flags)} page(s) flagged[/dim]")

    report: list[dict] = [{"page_no": p, "flags": flags[p]} for p in sorted(flags)]

    if not llm:
        if output:
            output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
            rprint(f"\n[green]Saved heuristic report → {output}[/green]")
        rprint("\n[dim]Pass --llm (with --pdf) to also verify these pages against the source image.[/dim]")
        raise typer.Exit(0)

    if not pdf:
        rprint("\n[red]--pdf is required when --llm is passed.[/red]")
        raise typer.Exit(1)
    pdf = _resolve_input(pdf)
    if not pdf.exists():
        rprint(f"[red]PDF not found: {pdf}[/red]")
        raise typer.Exit(1)

    n_checking = min(len(flags), max_pages) if max_pages else len(flags)
    console.print(f"\n[bold cyan]Verifying {n_checking} flagged page(s) with {model}…[/bold cyan]\n")

    try:
        results = verify_document(
            raw,
            pdf,
            flags=flags,
            api_key=api_key,
            model=model,
            base_url=base_url,
            max_pages=max_pages,
        )
    except ValueError as exc:
        rprint(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    rtbl = Table(title="Vision Verification", show_header=True, header_style="bold magenta")
    rtbl.add_column("Page", style="cyan")
    rtbl.add_column("Status")
    rtbl.add_column("Issues", justify="right")
    for r in results:
        style = {"ok": "green", "issues_found": "red"}.get(r.status, "yellow")
        rtbl.add_row(str(r.page_no), f"[{style}]{r.status}[/{style}]", str(len(r.issues)))
    console.print(rtbl)

    for r in results:
        if r.issues:
            rprint(f"\n[bold red]Page {r.page_no}:[/bold red]")
            for issue in r.issues:
                desc = issue.get("description", str(issue)) if isinstance(issue, dict) else str(issue)
                rprint(f"  • {desc}")
        elif r.status == "error":
            rprint(f"\n[yellow]Page {r.page_no}: verification error — {r.raw_response}[/yellow]")

    if output:
        full_report = [
            {
                "page_no": r.page_no,
                "flags": r.flags,
                "status": r.status,
                "issues": r.issues,
            }
            for r in results
        ]
        output.write_text(json.dumps(full_report, indent=2, ensure_ascii=False), encoding="utf-8")
        rprint(f"\n[green]Saved verification report → {output}[/green]")


if __name__ == "__main__":
    app()
