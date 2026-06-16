"""
examples/batch_convert.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
Batch convert all PDFs in the current directory using the ACCURATE profile.
"""

from pathlib import Path
from parsling import PdfParser, DocExporter

FOLDER = Path(".")     # change to your folder
OUTPUT = Path("./output")

parser = PdfParser(profile="accurate")

results = list(parser.parse_folder(FOLDER))
print(f"Found {len(results)} PDF(s)\n")

for result in results:
    if not result.ok:
        print(f"  FAIL  {result.source.name}: {result.error}")
        continue

    exp = DocExporter(result.document)
    out = exp.save(
        output_dir=OUTPUT / result.source.stem,
        stem=result.source.stem,
        formats=["md", "json", "csv"],
        save_figures=True,
    )
    tables = exp.to_tables()
    print(f"  OK    {result.source.name} → {out}  ({len(tables)} tables)")
