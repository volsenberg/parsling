"""
parsling/exporters.py
~~~~~~~~~~~~~~~~~~~~~
DocExporter — a unified export helper wrapping DoclingDocument's export API.

Supports:
    - Markdown
    - Rich Markdown (metadata-annotated, section-aware, table-caption linked)
    - JSON (lossless serialisation)
    - HTML (with optional embedded images)
    - Pandas DataFrames (one per table)
    - CSV files (one per table, saved to disk)
    - Figure images (saved to disk as PNGs)
    - RAG chunks via HybridChunker
    - save() — write all selected formats to an output directory in one call
"""

from __future__ import annotations

import json
import logging
import re
import textwrap
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from docling_core.types.doc import DoclingDocument

logger = logging.getLogger(__name__)

try:
    from docling.chunking import HybridChunker
    _HAS_CHUNKER = True
except ImportError:
    _HAS_CHUNKER = False


class DocExporter:
    """
    Export a ``DoclingDocument`` to multiple formats.

    Parameters
    ----------
    doc:
        The ``DoclingDocument`` to export.

    Examples
    --------
    ::

        from parsling import PdfParser, DocExporter

        doc = PdfParser().parse("report.pdf")
        exp = DocExporter(doc)

        print(exp.to_markdown())
        exp.save(Path("./output/"), formats=["md", "json", "csv"])
    """

    def __init__(self, doc: DoclingDocument) -> None:
        self.doc = doc

    # ------------------------------------------------------------------
    # Text formats
    # ------------------------------------------------------------------

    def to_markdown(self, image_mode: str = "placeholder") -> str:
        """
        Export to Markdown.

        Parameters
        ----------
        image_mode:
            ``"placeholder"`` — insert a text placeholder for figures.
            ``"embedded"`` — base64-embed figure images inline (large output).
        """
        return self.doc.export_to_markdown(image_mode=image_mode)

    def to_rich_markdown(
        self,
        skip_noise: bool = True,
        include_pictures: bool = True,
        caption_window: int = 4,
        min_text_len: int = 5,
        merge_split_tables: bool = True,
    ) -> str:
        """
        Export to metadata-rich Markdown.

        Unlike ``to_markdown()``, this method walks ``body.children`` in
        document order and annotates each element with HTML comments that
        record page number, section breadcrumb, and table/figure metadata.
        Section headers are tracked as a breadcrumb so every table and
        paragraph carries full context about where it lives in the document.

        Parameters
        ----------
        skip_noise:
            Skip ``page_header`` and ``page_footer`` text items (default True).
        include_pictures:
            Emit a placeholder comment for picture elements (default True).
        caption_window:
            How many body positions before/after a table to search for an
            adjacent ``caption`` text item (default 4).
        min_text_len:
            Minimum character length for a text item to be emitted; shorter
            fragments (OCR noise, single punctuation) are silently dropped
            (default 5).
        merge_split_tables:
            Automatically detect tables that Docling split across page
            boundaries and merge them into a single GFM table (default True).
            Repeated header rows on continuation pages are dropped. Set to
            False to keep each page fragment as a separate table.
        """
        raw = json.loads(self.doc.model_dump_json())
        return _build_rich_markdown(
            raw,
            skip_noise=skip_noise,
            include_pictures=include_pictures,
            caption_window=caption_window,
            min_text_len=min_text_len,
            merge_split_tables=merge_split_tables,
        )

    def to_json(self, indent: int = 2) -> str:
        """Lossless JSON serialisation of the DoclingDocument."""
        return self.doc.model_dump_json(indent=indent)

    def to_dict(self) -> dict:
        """Return the document as a plain Python dict (JSON-serialisable)."""
        return self.doc.model_dump()

    def to_html(self, embed_images: bool = False) -> str:
        """
        Export to HTML.

        Parameters
        ----------
        embed_images:
            If True, figure images are base64-embedded in the HTML output.
        """
        image_mode = "embedded" if embed_images else "placeholder"
        return self.doc.export_to_html(image_mode=image_mode)

    def to_doctags(self) -> str:
        """Export to DocTags format."""
        return self.doc.export_to_doctags()

    # ------------------------------------------------------------------
    # Tables
    # ------------------------------------------------------------------

    def to_tables(self) -> list[pd.DataFrame]:
        """
        Return every detected table as a Pandas DataFrame.

        Returns
        -------
        list[pd.DataFrame]
            One DataFrame per table, in document order.
        """
        frames = []
        for table in self.doc.tables:
            try:
                df = table.export_to_dataframe()
                frames.append(df)
            except Exception as exc:
                logger.warning("Could not export table to DataFrame: %s", exc)
        return frames

    # ------------------------------------------------------------------
    # RAG chunks
    # ------------------------------------------------------------------

    def to_chunks(
        self,
        tokenizer: str = "jinaai/jina-embeddings-v3",
        contextualize: bool = True,
    ) -> list[str]:
        """
        Split the document into RAG-ready text chunks using HybridChunker.

        Parameters
        ----------
        tokenizer:
            HuggingFace model name or OpenAI model name used to count tokens.
        contextualize:
            If True, prepend heading breadcrumbs to each chunk text.

        Returns
        -------
        list[str]
            List of chunk strings ready for embedding.

        Raises
        ------
        ImportError
            If ``docling-core[chunking]`` is not installed.
        """
        if not _HAS_CHUNKER:
            raise ImportError(
                "Chunking support requires: pip install 'docling-core[chunking]'"
            )
        chunker = HybridChunker(tokenizer=tokenizer)
        chunks = list(chunker.chunk(self.doc))
        if contextualize:
            return [chunker.contextualize(c) for c in chunks]
        return [c.text for c in chunks]

    # ------------------------------------------------------------------
    # Save to disk
    # ------------------------------------------------------------------

    def save(
        self,
        output_dir: str | Path,
        stem: str | None = None,
        formats: list[str] | None = None,
        save_figures: bool = True,
        rich_md_kwargs: dict | None = None,
    ) -> Path:
        """
        Write exports to ``output_dir`` in one call.

        Parameters
        ----------
        output_dir:
            Directory to write output files. Created if it does not exist.
        stem:
            Base filename (without extension). Defaults to the document
            ``origin.filename`` stem, or ``"document"`` if unavailable.
        formats:
            List of format strings to export. Any combination of:
            ``"md"``, ``"rich_md"``, ``"json"``, ``"html"``, ``"csv"``, ``"doctags"``.
            ``"rich_md"`` emits metadata-annotated Markdown via ``to_rich_markdown()``.
            Default: ``["json"]``.
        save_figures:
            If True and the document contains picture images, save them
            as PNGs to an ``images/`` sub-directory.

        Returns
        -------
        Path
            The resolved ``output_dir`` path.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if formats is None:
            formats = ["json"]

        # Resolve the output filename stem
        if stem is None:
            try:
                stem = Path(self.doc.origin.filename).stem
            except Exception:
                stem = "document"

        written: list[Path] = []

        if "md" in formats:
            p = output_dir / f"{stem}.md"
            p.write_text(self.to_markdown(), encoding="utf-8")
            written.append(p)
            logger.info("Saved Markdown → %s", p)

        if "rich_md" in formats:
            p = output_dir / f"{stem}_rich.md"
            p.write_text(self.to_rich_markdown(**(rich_md_kwargs or {})), encoding="utf-8")
            written.append(p)
            logger.info("Saved Rich Markdown → %s", p)

        if "json" in formats:
            p = output_dir / f"{stem}.json"
            # Use save_as_json for correct DoclingDocument serialisation
            # (handles image refs, embedded figures, etc.)
            self.doc.save_as_json(p)
            written.append(p)
            logger.info("Saved JSON → %s", p)

        if "html" in formats:
            p = output_dir / f"{stem}_html.md"
            html_content = self.to_html()
            p.write_text(f"```html\n{html_content}\n```", encoding="utf-8")
            written.append(p)
            logger.info("Saved HTML.md → %s", p)

        if "doctags" in formats:
            p = output_dir / f"{stem}_doctags.md"
            doctags_content = self.to_doctags()
            p.write_text(f"```xml\n{doctags_content}\n```", encoding="utf-8")
            written.append(p)
            logger.info("Saved DocTags.md → %s", p)

        if "csv" in formats:
            tables = self.to_tables()
            if tables:
                csv_dir = output_dir / "tables"
                csv_dir.mkdir(exist_ok=True)
                for i, df in enumerate(tables):
                    p = csv_dir / f"{stem}_table_{i + 1:02d}.csv"
                    df.to_csv(p, index=False, encoding="utf-8")
                    written.append(p)
                logger.info("Saved %d CSV tables → %s/", len(tables), csv_dir)
            else:
                logger.info("No tables found — skipping CSV export.")

        if save_figures:
            self._save_figures(output_dir, stem)

        return output_dir

    def _save_figures(self, output_dir: Path, stem: str) -> None:
        """Save extracted figure images as PNGs to output_dir/images/."""
        pictures = list(self.doc.pictures)
        if not pictures:
            return

        img_dir = output_dir / "images"
        img_dir.mkdir(exist_ok=True)
        saved = 0

        for i, picture in enumerate(pictures):
            try:
                img = picture.get_image(self.doc)
                if img is None:
                    continue
                # Annotate filename with classification label if available
                label = "figure"
                if picture.annotations:
                    label = picture.annotations[0].predicted_class or "figure"
                    label = label.lower().replace(" ", "_")
                p = img_dir / f"{stem}_{label}_{i + 1:02d}.png"
                img.save(p, format="PNG")
                saved += 1
            except Exception as exc:
                logger.warning("Could not save figure %d: %s", i + 1, exc)

        if saved:
            logger.info("Saved %d figure image(s) → %s/", saved, img_dir)


# ---------------------------------------------------------------------------
# Rich Markdown builder — module-level helper
# ---------------------------------------------------------------------------

_NOISE_LABELS = {"page_header", "page_footer"}
_SKIP_TABLE_LABELS = {"document_index"}


def _build_rich_markdown(
    raw: dict,
    skip_noise: bool = True,
    include_pictures: bool = True,
    caption_window: int = 4,
    min_text_len: int = 5,
    merge_split_tables: bool = True,
) -> str:
    """Walk a raw DoclingDocument dict and emit metadata-annotated Markdown."""

    # Build a flat ref → item lookup so $ref pointers resolve in O(1).
    ref_map: dict[str, dict] = {}
    for collection in ("texts", "tables", "pictures", "groups"):
        for item in raw.get(collection, []):
            ref_map[item["self_ref"]] = item

    # Ordered list of body refs (preserves reading order).
    # Raw JSON uses "$ref"; after model_dump_json() round-trip Docling uses "cref".
    body_refs: list[str] = [
        c.get("$ref") or c.get("cref", "")
        for c in raw.get("body", {}).get("children", [])
    ]

    # Pre-index caption positions for fast window lookups.
    caption_positions: dict[int, str] = {}
    for pos, ref in enumerate(body_refs):
        item = ref_map.get(ref, {})
        if item.get("label") == "caption":
            caption_positions[pos] = item.get("text", "")

    # -----------------------------------------------------------------------
    # Pre-pass: detect and merge page-split tables
    # -----------------------------------------------------------------------
    # merged_grids: ref → merged grid (list of rows) replacing the original
    # absorbed_refs: refs of continuation fragments to skip during main walk
    merged_grids: dict[str, list] = {}
    merged_pages: dict[str, list[int]] = {}   # ref → all pages spanned
    absorbed_refs: set[str] = set()

    if merge_split_tables:
        _detect_and_merge(body_refs, ref_map, merged_grids, merged_pages, absorbed_refs)

    # -----------------------------------------------------------------------
    # Frontmatter
    # -----------------------------------------------------------------------
    origin = raw.get("origin", {})
    pages = raw.get("pages", {})
    page_count = len(pages) if isinstance(pages, dict) else 0

    lines: list[str] = [
        "---",
        f"document: {origin.get('filename', raw.get('name', 'unknown'))}",
        f"schema: DoclingDocument v{raw.get('version', '?')}",
        f"pages: {page_count}",
        f"exported: {date.today().isoformat()}",
        "---",
        "",
    ]

    # -----------------------------------------------------------------------
    # Walk body in order
    # -----------------------------------------------------------------------
    section_stack: list[str] = []
    table_counter = 0

    def _breadcrumb() -> str:
        return " > ".join(s for s in section_stack if s)

    def _page(item: dict) -> int | str:
        prov = item.get("prov") or []
        return prov[0].get("page_no", "?") if prov else "?"

    def _find_caption(pos: int) -> str | None:
        for delta in range(1, caption_window + 1):
            for candidate in (pos - delta, pos + delta):
                if candidate in caption_positions:
                    return caption_positions[candidate]
        return None

    def _render_grid(grid: list) -> str:
        if not grid:
            return ""
        rows: list[str] = []
        for r, row in enumerate(grid):
            cells = [cell.get("text", "").replace("|", "\\|").replace("\n", " ") for cell in row]
            rows.append("| " + " | ".join(cells) + " |")
            if r == 0:
                rows.append("| " + " | ".join("---" for _ in row) + " |")
        return "\n".join(rows)

    for pos, ref in enumerate(body_refs):
        item = ref_map.get(ref)
        if item is None:
            continue

        # Skip continuation fragments that were merged into their head table.
        if ref in absorbed_refs:
            continue

        label = item.get("label", "")
        page = _page(item)

        # --- Noise ---
        if skip_noise and label in _NOISE_LABELS:
            continue

        # --- Section headers ---
        if label == "section_header":
            text = item.get("text", "").strip()
            if not text:
                continue
            depth = _heading_depth(text, section_stack)
            section_stack = section_stack[: depth - 1] + [text]
            hashes = "#" * min(depth + 1, 6)
            lines.append(f"\n{hashes} {text}")
            lines.append(f"<!-- section page={page} depth={depth} -->")
            continue

        # --- Caption (rendered inline next to its table/figure) ---
        if label == "caption":
            continue

        # --- Tables ---
        if label == "table":
            if item.get("label") in _SKIP_TABLE_LABELS:
                continue
            table_counter += 1

            # Use merged grid if available, else the original.
            if ref in merged_grids:
                grid = merged_grids[ref]
                all_pages = merged_pages.get(ref, [page])
                page_str = "-".join(str(p) for p in all_pages)
                num_rows = len(grid)
                num_cols = len(grid[0]) if grid else "?"
                merge_note = f", merged_from_pages={page_str}"
            else:
                grid = (item.get("data") or {}).get("grid", [])
                num_rows = (item.get("data") or {}).get("num_rows", "?")
                num_cols = (item.get("data") or {}).get("num_cols", "?")
                page_str = str(page)
                merge_note = ""

            caption = _find_caption(pos) or ""
            breadcrumb = _breadcrumb()
            lines.append("")
            lines.append(
                f"<!-- TABLE {table_counter}: page={page_str}, rows={num_rows}, cols={num_cols}"
                + (f', section="{breadcrumb}"' if breadcrumb else "")
                + (f', caption="{caption}"' if caption else "")
                + merge_note
                + " -->"
            )
            if caption:
                lines.append(f"**{caption}**")
                lines.append("")
            md_table = _render_grid(grid)
            if md_table:
                lines.append(md_table)
            else:
                lines.append("<!-- table grid not available -->")
            lines.append("")
            continue

        # --- Pictures ---
        if label == "picture":
            if not include_pictures:
                continue
            caption = _find_caption(pos) or ""
            breadcrumb = _breadcrumb()
            lines.append("")
            lines.append(
                "<!-- FIGURE: page="
                + str(page)
                + (f', section="{breadcrumb}"' if breadcrumb else "")
                + (f', caption="{caption}"' if caption else "")
                + " -->"
            )
            if caption:
                lines.append(f"*{caption}*")
                lines.append("")
            continue

        # --- List items ---
        if label == "list_item":
            text = item.get("text", "").strip()
            if text:
                lines.append(f"- {text}")
            continue

        # --- Footnotes ---
        if label == "footnote":
            text = item.get("text", "").strip()
            if text:
                lines.append(f"\n> [^fn] {text}")
            continue

        # --- Body text ---
        if label == "text":
            text = item.get("text", "").strip()
            if not text or len(text) < min_text_len:
                continue
            breadcrumb = _breadcrumb()
            lines.append("")
            lines.append(text)
            lines.append(
                "<!-- text page="
                + str(page)
                + (f' section="{breadcrumb}"' if breadcrumb else "")
                + " -->"
            )
            continue

        # --- key_value_area groups (pass-through) ---
        if label == "key_value_area":
            text = item.get("text", "").strip()
            if text:
                lines.append(f"\n> {text}")

    return "\n".join(lines) + "\n"


def _detect_and_merge(
    body_refs: list[str],
    ref_map: dict[str, dict],
    merged_grids: dict[str, list],
    merged_pages: dict[str, list[int]],
    absorbed_refs: set[str],
) -> None:
    """
    Scan body_refs for consecutive real tables on adjacent pages with the same
    column count and merge their grids in-place into merged_grids.

    A continuation table's first row is dropped if it is identical to the
    head table's first row (repeated page header pattern).
    """
    # Collect (body_pos, ref, item) for every real table in body order.
    table_entries: list[tuple[int, str, dict]] = []
    for pos, ref in enumerate(body_refs):
        item = ref_map.get(ref, {})
        if item.get("label") == "table" and item.get("label") not in _SKIP_TABLE_LABELS:
            table_entries.append((pos, ref, item))

    def _get_page(item: dict) -> int | None:
        prov = item.get("prov") or []
        v = prov[0].get("page_no") if prov else None
        return int(v) if v is not None else None

    def _row_texts(row: list) -> list[str]:
        return [cell.get("text", "").strip() for cell in row]

    i = 0
    while i < len(table_entries):
        head_pos, head_ref, head_item = table_entries[i]
        head_page = _get_page(head_item)
        head_cols = (head_item.get("data") or {}).get("num_cols", -1)
        head_grid = list((head_item.get("data") or {}).get("grid", []))
        head_header = _row_texts(head_grid[0]) if head_grid else []

        chain_pages = [head_page] if head_page is not None else []
        chain_absorbed: list[str] = []

        j = i + 1
        while j < len(table_entries):
            _, cont_ref, cont_item = table_entries[j]
            cont_page = _get_page(cont_item)
            cont_cols = (cont_item.get("data") or {}).get("num_cols", -1)
            cont_grid = list((cont_item.get("data") or {}).get("grid", []))

            # Chain continues only if: adjacent page, same column count.
            prev_page = chain_pages[-1] if chain_pages else head_page
            if (
                cont_page is None
                or prev_page is None
                or cont_page - prev_page != 1
                or cont_cols != head_cols
                or not cont_grid
            ):
                break

            # Drop the continuation's first row if it repeats the head header
            # (page-repeated column header) or repeats itself (sub-header rows).
            cont_rows = cont_grid
            if cont_rows and _row_texts(cont_rows[0]) == head_header:
                cont_rows = cont_rows[1:]

            head_grid = head_grid + cont_rows
            chain_pages.append(cont_page)
            chain_absorbed.append(cont_ref)
            j += 1

        if chain_absorbed:
            # Store merged result; mark continuations absorbed.
            merged_grids[head_ref] = head_grid
            merged_pages[head_ref] = chain_pages
            absorbed_refs.update(chain_absorbed)

        i = j if chain_absorbed else i + 1


def _heading_depth(text: str, stack: list[str]) -> int:
    """
    Infer Markdown heading depth (1-based) from the text content.

    Priority order:
    1. Leading numeric outline prefix (e.g. "1.2.3 Title" → depth 3).
    2. ALL-CAPS short text with no sub-section below → treat as depth 1.
    3. Fall back to current stack depth + 1 (continuation), capped at 4.
    """
    m = re.match(r"^(\d+(?:\.\d+)*)\s", text)
    if m:
        return len(m.group(1).split("."))
    if text == text.upper() and len(text) < 60:
        return 1
    return min(len(stack) + 1, 4)
