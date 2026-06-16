"""
parsling
~~~~~~~~
A robust PDF parser built on Docling for complex financial
and government report layouts.

Quick start::

    from parsling import PdfParser, DocExporter

    # Parse with the ACCURATE profile (default)
    parser = PdfParser()
    doc = parser.parse("report.pdf")

    # Export
    exp = DocExporter(doc)
    exp.save(Path("./output/"), formats=["md", "json", "csv"])

Profiles::

    PdfParser(profile="fast")     # Tesseract + TableFormer FAST
    PdfParser(profile="accurate") # EasyOCR + TableFormer ACCURATE + enrichments
    PdfParser(profile="vlm")      # Granite-Docling-258M end-to-end VLM
"""

from parsling.config import ParseProfile
from parsling.converter import ParseResult, PdfParser
from parsling.exporters import DocExporter
from parsling.profiles import ACCURATE, FAST, VLM

__all__ = [
    "PdfParser",
    "ParseResult",
    "ParseProfile",
    "DocExporter",
    "FAST",
    "ACCURATE",
    "VLM",
]

__version__ = "0.1.0"
