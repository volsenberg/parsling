"""
parsling/config.py
~~~~~~~~~~~~~~~~~~
ParseProfile dataclass — defines every knob for the parsing pipeline.
When `use_vlm=True`, all StandardPdfPipeline fields are ignored and
Granite-Docling-258M handles the entire conversion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ParseProfile:
    """
    Configuration for a single parsing run.

    Parameters
    ----------
    use_vlm:
        If True, use VlmPipeline with Granite-Docling-258M.
        All fields below are ignored when this is True — the VLM
        replaces the entire pipeline (layout + OCR + table + enrichment).

    do_ocr:
        Enable OCR. Required for scanned/image-based PDFs.

    do_table_structure:
        Enable TableFormer table structure recognition.

    table_mode:
        ``"fast"`` — higher throughput, lower precision (simple tables).
        ``"accurate"`` — best structural recognition for complex/financial tables.

    do_picture_classification:
        Enable DocumentFigureClassifier-v2.5. Classifies detected figures
        into categories: Chart, Diagram, Natural Image, Table, etc.

    do_chart_extraction:
        Convert detected chart images into tabular data (requires
        ``do_picture_classification=True``).

    do_formula_enrichment:
        Enable CodeFormulaV2 formula enrichment — outputs LaTeX for
        detected mathematical expressions.

    do_code_enrichment:
        Enable CodeFormulaV2 code enrichment — identifies programming
        language of detected code blocks.

    generate_page_images:
        Render full-page images into the DoclingDocument. Increases
        memory usage significantly — disable unless needed.

    generate_picture_images:
        Extract cropped figure/picture images into the DoclingDocument.
        Required for saving figures to disk in the exporter.

    images_scale:
        Resolution multiplier for rendered images (1.0 = 72 dpi,
        2.0 = 144 dpi). Higher = better OCR/VLM accuracy, more memory.

    ocr_engine:
        ``"easyocr"`` — multi-language, GPU-capable (default for ACCURATE).
        ``"tesseract_cli"`` — lightweight CPU-only (default for FAST).

    ocr_langs:
        Language codes for the chosen OCR engine.
        EasyOCR:   ``["en", "id"]``
        Tesseract: ``["eng", "ind"]``

    tesseract_psm:
        Tesseract Page Segmentation Mode. Only applies to ``tesseract_cli``.
        Common values:
        ``3``  — Fully automatic (default Tesseract, uses OSD).
        ``6``  — Assume a single uniform block of text (no OSD, **recommended**).
        ``11`` — Sparse text — good for documents with scattered text.

    accelerator:
        Target device for model inference.
        ``"cpu"`` | ``"cuda"`` | ``"mps"`` (Apple Silicon)

    num_threads:
        Number of CPU threads for model inference.

    artifacts_path:
        Optional path to a local directory with pre-downloaded model
        weights. Set this for air-gapped / offline environments.
    """

    # VLM switch
    use_vlm: bool = False

    # OCR
    do_ocr: bool = True
    ocr_engine: Literal["easyocr", "tesseract_cli"] = "easyocr"
    ocr_langs: list[str] = field(default_factory=lambda: ["en", "id"])
    tesseract_psm: int = 6  # 6 = uniform text block, no OSD (avoids OSD errors on sparse pages)

    # Table
    do_table_structure: bool = True
    table_mode: Literal["fast", "accurate"] = "accurate"

    # Enrichments
    do_picture_classification: bool = True
    do_chart_extraction: bool = True
    chart_extraction_model: Literal["granite-vision", "granite-vision-v4"] = "granite-vision"
    do_formula_enrichment: bool = True
    do_code_enrichment: bool = True

    # Image generation
    generate_page_images: bool = False
    generate_picture_images: bool = True
    images_scale: float = 2.0

    # Hardware
    accelerator: Literal["cpu", "cuda", "mps"] = "cpu"
    num_threads: int = 4

    # Offline model cache
    artifacts_path: str | None = None
