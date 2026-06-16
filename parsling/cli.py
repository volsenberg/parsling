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

import logging
import logging.handlers
import queue
import re
import threading
from pathlib import Path
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

app = typer.Typer(
    name="parsling",
    help="PDF parser built on Docling — for complex financial and government report layouts.",
    add_completion=False,
)
console = Console()

_PROFILE_CHOICES = ["fast", "accurate", "vlm"]
_FORMAT_CHOICES = ["md", "rich_md", "json", "html", "csv", "doctags"]

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

    if path.suffix.lower() == ".json":
        console.print(f"\n[bold cyan]parsling[/bold cyan] · loading structured JSON · {path.name}\n")
        from docling_core.types.doc import DoclingDocument
        with console.status(f"[bold green]Loading {path.name}…"):
            doc = DoclingDocument.load_from_json(path)
    else:
        console.print(f"\n[bold cyan]parsling[/bold cyan] · profile=[yellow]{profile}[/yellow] · {path.name}\n")
        total_pages = _count_pdf_pages(path)
        if total_pages:
            console.print(f"[dim]  {total_pages} pages detected[/dim]\n")
        parser = PdfParser(profile=profile)
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

    console.print(f"\n[bold cyan]parsling batch[/bold cyan] · profile=[yellow]{profile}[/yellow] · {folder}\n")

    parser = PdfParser(profile=profile)
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


if __name__ == "__main__":
    app()
