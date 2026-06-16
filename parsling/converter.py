"""
parsling/converter.py
~~~~~~~~~~~~~~~~~~~~~
Core PdfParser class — the main entry point for converting PDF files
into DoclingDocument objects using Docling's pipeline infrastructure.

Supported profiles (by name string or ParseProfile instance):
    "fast"     → Tesseract + TableFormer FAST, no enrichments
    "accurate" → EasyOCR [en,id] + TableFormer ACCURATE + all enrichments (default)
    "vlm"      → Granite-Docling-258M end-to-end VLM pipeline
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import (
    EasyOcrOptions,
    PdfPipelineOptions,
    TableFormerMode,
    TesseractCliOcrOptions,
    VlmPipelineOptions,
)
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.chart_extraction_options import (
    ChartExtractionModelKind,
    ChartExtractionModelOptions,
)
from docling.datamodel import vlm_model_specs
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline
from docling_core.types.doc import DoclingDocument

from parsling.config import ParseProfile
from parsling.profiles import ACCURATE, REGISTRY

logger = logging.getLogger(__name__)


@dataclass
class ParseResult:
    """Wraps a single conversion result with status metadata."""

    document: DoclingDocument | None
    """The parsed DoclingDocument. None if conversion failed."""

    source: Path
    """Original input file path."""

    status: ConversionStatus
    """SUCCESS, PARTIAL_SUCCESS, or FAILURE."""

    error: str | None = None
    """Error message if status is FAILURE."""

    @property
    def ok(self) -> bool:
        """True if conversion succeeded (fully or partially)."""
        return self.status in (
            ConversionStatus.SUCCESS,
            ConversionStatus.PARTIAL_SUCCESS,
        )


def _resolve_profile(profile: ParseProfile | str) -> ParseProfile:
    """Resolve a profile name string or ParseProfile instance."""
    if isinstance(profile, str):
        key = profile.lower()
        if key not in REGISTRY:
            raise ValueError(
                f"Unknown profile {profile!r}. "
                f"Available: {list(REGISTRY.keys())}"
            )
        return REGISTRY[key]
    return profile


def _build_accelerator(profile: ParseProfile) -> AcceleratorOptions:
    device_map = {
        "cpu": AcceleratorDevice.CPU,
        "cuda": AcceleratorDevice.CUDA,
        "mps": AcceleratorDevice.MPS,
    }
    return AcceleratorOptions(
        num_threads=profile.num_threads,
        device=device_map.get(profile.accelerator, AcceleratorDevice.CPU),
    )


def _build_standard_converter(profile: ParseProfile) -> DocumentConverter:
    """Build a DocumentConverter using StandardPdfPipeline."""
    pipeline_options = PdfPipelineOptions()

    # OCR
    pipeline_options.do_ocr = profile.do_ocr
    if profile.do_ocr:
        if profile.ocr_engine == "easyocr":
            pipeline_options.ocr_options = EasyOcrOptions(
                lang=profile.ocr_langs,
                use_gpu=(profile.accelerator != "cpu"),
                force_full_page_ocr=False,
            )
        else:
            pipeline_options.ocr_options = TesseractCliOcrOptions(
                lang=profile.ocr_langs,
                # psm=6 by default: "Assume a single uniform block of text"
                # Disables OSD (Orientation & Script Detection) which fails
                # on image-heavy / sparse pages with too few characters.
                psm=profile.tesseract_psm,
            )

    # Table structure
    pipeline_options.do_table_structure = profile.do_table_structure
    if profile.do_table_structure:
        pipeline_options.table_structure_options.mode = (
            TableFormerMode.ACCURATE
            if profile.table_mode == "accurate"
            else TableFormerMode.FAST
        )

    # Enrichments
    pipeline_options.do_picture_classification = profile.do_picture_classification
    pipeline_options.do_chart_extraction = profile.do_chart_extraction
    if profile.do_chart_extraction:
        model_kind = (
            ChartExtractionModelKind.GRANITE_VISION
            if profile.chart_extraction_model == "granite-vision"
            else ChartExtractionModelKind.GRANITE_VISION_V4
        )
        pipeline_options.chart_extraction_options = ChartExtractionModelOptions(
            model=model_kind
        )
    pipeline_options.do_formula_enrichment = profile.do_formula_enrichment
    pipeline_options.do_code_enrichment = profile.do_code_enrichment

    # Image generation
    pipeline_options.generate_page_images = profile.generate_page_images
    pipeline_options.generate_picture_images = profile.generate_picture_images
    pipeline_options.images_scale = profile.images_scale

    # Hardware
    pipeline_options.accelerator_options = _build_accelerator(profile)

    # Offline model cache
    if profile.artifacts_path:
        pipeline_options.artifacts_path = profile.artifacts_path

    return DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        },
    )


def _build_vlm_converter(profile: ParseProfile) -> DocumentConverter:
    """Build a DocumentConverter using VlmPipeline with Granite-Docling-258M."""
    pipeline_options = VlmPipelineOptions(
        vlm_options=vlm_model_specs.GRANITEDOCLING_TRANSFORMERS,
    )
    pipeline_options.accelerator_options = _build_accelerator(profile)

    return DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=VlmPipeline,
                pipeline_options=pipeline_options,
            )
        },
    )


class PdfParser:
    """
    High-level PDF parser wrapping Docling's DocumentConverter.

    Parameters
    ----------
    profile:
        A ``ParseProfile`` instance or one of the preset name strings:
        ``"fast"``, ``"accurate"`` (default), ``"vlm"``.

    Examples
    --------
    Basic usage::

        from parsling import PdfParser

        parser = PdfParser()                         # ACCURATE profile
        doc = parser.parse("report.pdf")
        print(doc.export_to_markdown())

    VLM profile::

        parser = PdfParser(profile="vlm")
        doc = parser.parse("complex_layout.pdf")

    Batch folder::

        results = list(parser.parse_folder(Path("./pdfs/")))
    """

    def __init__(self, profile: ParseProfile | str = "accurate") -> None:
        self.profile = _resolve_profile(profile)
        self._converter = (
            _build_vlm_converter(self.profile)
            if self.profile.use_vlm
            else _build_standard_converter(self.profile)
        )
        logger.info(
            "PdfParser initialised — pipeline=%s",
            "VlmPipeline (Granite-Docling-258M)" if self.profile.use_vlm else "StandardPdfPipeline",
        )

    # ------------------------------------------------------------------
    # Single document
    # ------------------------------------------------------------------

    def parse(
        self,
        path: str | Path,
        page_range: tuple[int, int] | None = None,
    ) -> DoclingDocument:
        """
        Parse a single PDF file and return its ``DoclingDocument``.

        Parameters
        ----------
        path:
            Path to the PDF file (local) or a URL.
        page_range:
            Optional ``(start, end)`` tuple (1-indexed, inclusive) to
            parse only a subset of pages.

        Returns
        -------
        DoclingDocument

        Raises
        ------
        RuntimeError
            If Docling reports a FAILURE status.
        """
        path = Path(path) if not isinstance(path, str) else path
        kwargs: dict = {}
        if page_range is not None:
            kwargs["page_range"] = page_range
            logger.info("Parsing pages %s-%s of %s using Docling", page_range[0], page_range[1], path.name)
        else:
            logger.info("Parsing entire document: %s using Docling", path.name)

        result = self._converter.convert(str(path), **kwargs)
        logger.info("Docling parsing completed with status: %s", result.status)

        if result.status == ConversionStatus.FAILURE:
            raise RuntimeError(
                f"Failed to convert {path!r}. "
                "Check that the file is a valid PDF."
            )

        if result.status == ConversionStatus.PARTIAL_SUCCESS:
            logger.warning("Partial conversion for %s — some pages may be missing.", path)

        return result.document

    # ------------------------------------------------------------------
    # Batch — list of files
    # ------------------------------------------------------------------

    def parse_many(
        self,
        paths: list[str | Path],
    ) -> Iterator[ParseResult]:
        """
        Convert a list of PDF files, yielding ``ParseResult`` objects.

        Uses Docling's internal batching iterator to avoid memory spikes.
        Failed conversions are yielded (not raised) so the batch continues.

        Parameters
        ----------
        paths:
            List of file paths or URLs.

        Yields
        ------
        ParseResult
        """
        str_paths = [str(p) for p in paths]
        for result in self._converter.convert_all(str_paths):
            if result.status == ConversionStatus.FAILURE:
                logger.error("Conversion failed for %s", result.input.file)
                yield ParseResult(
                    document=None,
                    source=Path(result.input.file),
                    status=result.status,
                    error="Docling reported FAILURE status.",
                )
            else:
                yield ParseResult(
                    document=result.document,
                    source=Path(result.input.file),
                    status=result.status,
                )

    # ------------------------------------------------------------------
    # Batch — folder glob
    # ------------------------------------------------------------------

    def parse_folder(
        self,
        folder: str | Path,
        glob: str = "**/*.pdf",
    ) -> Iterator[ParseResult]:
        """
        Discover and convert all PDFs in a folder matching ``glob``.

        Parameters
        ----------
        folder:
            Root directory to search.
        glob:
            Glob pattern relative to ``folder``. Default: ``**/*.pdf``
            (recursive).

        Yields
        ------
        ParseResult
        """
        folder = Path(folder)
        paths = list(folder.glob(glob))
        if not paths:
            logger.warning("No PDFs found in %s with pattern %r", folder, glob)
            return
        logger.info("Found %d PDFs in %s", len(paths), folder)
        yield from self.parse_many(paths)
