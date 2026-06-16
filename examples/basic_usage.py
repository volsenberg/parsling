"""
examples/basic_usage.py
~~~~~~~~~~~~~~~~~~~~~~~
Minimal example: parse a PDF and print the Markdown output.
"""

from pathlib import Path
from parsling import PdfParser, DocExporter

# Default profile is ACCURATE (EasyOCR + TableFormer ACCURATE + enrichments)
parser = PdfParser()

doc = parser.parse(Path("01_aceh.pdf"))

print("=== Markdown (first 2000 chars) ===")
md = DocExporter(doc).to_markdown()
print(md[:2000])

print(f"\n=== Stats ===")
print(f"Pages  : {len(list(doc.pages))}")
print(f"Tables : {len(doc.tables)}")
print(f"Figures: {len(list(doc.pictures))}")
