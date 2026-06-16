"""
parsling/profiles.py
~~~~~~~~~~~~~~~~~~~~
Ready-to-use ParseProfile presets.

    FAST     — Tesseract + TableFormer FAST. Quick preview, low resource.
    ACCURATE — EasyOCR [en,id] + TableFormer ACCURATE + all enrichments.
               This is the go-to profile for financial / government reports.
    VLM      — Granite-Docling-258M end-to-end. Replaces the entire pipeline.
               Best fallback for non-standard or unusual layouts.
"""

from parsling.config import ParseProfile

FAST = ParseProfile(
    use_vlm=False,
    do_ocr=True,
    ocr_engine="tesseract_cli",
    ocr_langs=["eng", "ind"],
    do_table_structure=True,
    table_mode="fast",
    do_picture_classification=False,
    do_chart_extraction=False,
    do_formula_enrichment=False,
    do_code_enrichment=False,
    generate_page_images=False,
    generate_picture_images=False,
    images_scale=1.0,
    accelerator="cpu",
    num_threads=4,
)

ACCURATE = ParseProfile(
    use_vlm=False,
    do_ocr=True,
    ocr_engine="easyocr",
    ocr_langs=["en", "id"],
    do_table_structure=True,
    table_mode="accurate",
    do_picture_classification=True,   # DocumentFigureClassifier-v2.5
    do_chart_extraction=True,         # chart images → tabular data
    do_formula_enrichment=True,       # CodeFormulaV2 → LaTeX
    do_code_enrichment=True,          # CodeFormulaV2 → language ID
    generate_page_images=False,
    generate_picture_images=True,
    images_scale=2.0,
    accelerator="cpu",
    num_threads=4,
)

VLM = ParseProfile(
    use_vlm=True,
    # All fields below are ignored by VlmPipeline —
    # Granite-Docling-258M handles layout, OCR, tables, and enrichment
    # internally as a single end-to-end pass.
    do_ocr=False,
    ocr_engine="easyocr",
    ocr_langs=["en", "id"],
    do_table_structure=False,
    table_mode="accurate",
    do_picture_classification=False,
    do_chart_extraction=False,
    do_formula_enrichment=False,
    do_code_enrichment=False,
    generate_page_images=False,
    generate_picture_images=False,
    images_scale=2.0,
    accelerator="cpu",
    num_threads=4,
)

# Registry — used by PdfParser to look up profiles by name string
REGISTRY: dict[str, ParseProfile] = {
    "fast": FAST,
    "accurate": ACCURATE,
    "vlm": VLM,
}
