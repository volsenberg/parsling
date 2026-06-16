"""
examples/financial_report.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Full financial report pipeline using the ACCURATE profile.
Saves Markdown, JSON, CSV tables, and figure images to ./output/.
"""

from pathlib import Path
from parsling import PdfParser, DocExporter

PDF = Path("01_aceh.pdf")
OUTPUT = Path("./output")

# ACCURATE: EasyOCR [en,id] + TableFormer ACCURATE
#           + DocumentFigureClassifier-v2.5
#           + CodeFormulaV2 (formula + code enrichment)
#           + chart extraction
parser = PdfParser(profile="accurate")

print(f"Parsing {PDF.name} …")
doc = parser.parse(PDF)

exp = DocExporter(doc)

# Save all formats + figures
out_dir = exp.save(
    output_dir=OUTPUT / PDF.stem,
    stem=PDF.stem,
    formats=["md", "json", "html", "csv"],
    save_figures=True,
)

print(f"\nDone! Output → {out_dir}")
print(f"  Pages  : {len(list(doc.pages))}")
print(f"  Tables : {len(doc.tables)}")
print(f"  Figures: {len(list(doc.pictures))}")

# Print each table as a DataFrame preview
tables = exp.to_tables()
for i, df in enumerate(tables):
    print(f"\n--- Table {i + 1} ({df.shape[0]}r × {df.shape[1]}c) ---")
    print(df.head(5).to_string(index=False))
