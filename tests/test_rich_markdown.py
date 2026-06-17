"""
tests/test_rich_markdown.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Regression tests for the rich-markdown recovery heuristics in
parsling/exporters.py, seeded from real extraction bugs found (and fixed)
on production documents:

    - PKSOP4D formula (2026 PMK 35, page 109): a heading line fused onto
      an unrelated formula fragment, and rendered in the wrong order.
    - "Pasal 117" breadcrumb growth: repeated heading patterns nesting
      one level deeper on every occurrence instead of staying flat.
    - "DHAHANA PUTRA" false positive (page 108): a legitimate single
      all-caps sentence incorrectly split mid-phrase, fusing a person's
      name with an unrelated trailing fragment.
    - List items silently dropped because they live nested inside a
      Docling "list" group rather than directly under body.children.

These use small synthetic DoclingDocument-shaped JSON fixtures rather
than full real documents, so they run fast and don't require bundling
proprietary PDFs as test fixtures — golden-file testing, but the "golden"
content is built in Python for each case instead of checked-in binaries.
"""

from parsling.exporters import _build_rich_markdown, _split_embedded_heading


def _text_item(ref_idx, text, label="text", marker=None, page=1, top=100.0, bottom=90.0):
    item = {
        "self_ref": f"#/texts/{ref_idx}",
        "label": label,
        "text": text,
        "prov": [{
            "page_no": page,
            "bbox": {"l": 0.0, "r": 500.0, "t": top, "b": bottom, "coord_origin": "BOTTOMLEFT"},
        }],
    }
    if marker is not None:
        item["marker"] = marker
    return item


def _group_item(ref_idx, children_refs, label="list"):
    return {
        "self_ref": f"#/groups/{ref_idx}",
        "label": label,
        "children": [{"$ref": r} for r in children_refs],
    }


def _raw_doc(texts=None, groups=None, tables=None, body_refs=None):
    return {
        "schema_name": "DoclingDocument",
        "version": "1.10.0",
        "name": "test",
        "origin": {"filename": "test.pdf"},
        "body": {"children": [{"$ref": r} for r in (body_refs or [])]},
        "groups": groups or [],
        "texts": texts or [],
        "pictures": [],
        "tables": tables or [],
        "pages": {"1": {}},
    }


def _heading_hashes(md: str, text: str) -> int:
    for line in md.splitlines():
        if line.lstrip("#").strip() == text:
            return len(line) - len(line.lstrip("#"))
    raise AssertionError(f"heading {text!r} not found in:\n{md}")


# -----------------------------------------------------------------------
# List-group flattening
# -----------------------------------------------------------------------

def test_list_items_nested_in_group_are_rendered():
    """List items live inside a 'list' group, not directly under body.children
    — a flat walk silently drops them. They must still appear in the output."""
    texts = [
        _text_item(0, "first item text", label="list_item", marker="a."),
        _text_item(1, "second item text", label="list_item", marker="b."),
    ]
    groups = [_group_item(0, ["#/texts/0", "#/texts/1"])]
    raw = _raw_doc(texts=texts, groups=groups, body_refs=["#/groups/0"])

    md = _build_rich_markdown(raw)

    assert "a. first item text" in md
    assert "b. second item text" in md


# -----------------------------------------------------------------------
# Heading depth stays flat for repeated patterns ("Pasal 117" breadcrumb bug)
# -----------------------------------------------------------------------

def test_repeated_heading_kind_stays_at_same_depth():
    """Pasal 1, Pasal 2, ... must render as siblings, not nest one level
    deeper on every occurrence (previously produced breadcrumbs like
    'Pasal 93 > Pasal 117' that grew unbounded)."""
    texts = [
        _text_item(0, "Pasal 1", label="section_header"),
        _text_item(1, "body text under pasal one", label="text"),
        _text_item(2, "Pasal 2", label="section_header"),
        _text_item(3, "body text under pasal two", label="text"),
    ]
    raw = _raw_doc(texts=texts, body_refs=[f"#/texts/{i}" for i in range(4)])

    md = _build_rich_markdown(raw)

    assert _heading_hashes(md, "Pasal 1") == _heading_hashes(md, "Pasal 2")


# -----------------------------------------------------------------------
# Page-split continuation merging
# -----------------------------------------------------------------------

def test_unmarked_continuation_fragment_merges_into_previous_item():
    """An unmarked list item that looks like a sentence fragment (lowercase
    start) is the tail of a page-split clause — it must reattach to the
    previous item instead of rendering as its own disconnected bullet."""
    texts = [
        _text_item(0, "first part of the sentence", label="list_item", marker="a."),
        _text_item(1, "continues here lowercase", label="list_item", marker=""),
    ]
    groups = [_group_item(0, ["#/texts/0"])]
    raw = _raw_doc(texts=texts, groups=groups, body_refs=["#/groups/0", "#/texts/1"])

    md = _build_rich_markdown(raw)

    assert "a. first part of the sentence continues here lowercase" in md
    assert "- continues here lowercase" not in md


# -----------------------------------------------------------------------
# Embedded-heading recovery (PKSOP4D bug)
# -----------------------------------------------------------------------

def test_split_embedded_heading_recovers_genuine_merge():
    heading, remainder = _split_embedded_heading(
        "RUMUS PERHITUNGAN NILAI KINERJA PKSOP4D) + (0,2 x Kepatuhan Pelaporan SPT)"
    )
    assert heading == "RUMUS PERHITUNGAN NILAI KINERJA"
    assert remainder == "PKSOP4D) + (0,2 x Kepatuhan Pelaporan SPT)"


def test_split_embedded_heading_ignores_legitimate_allcaps_titles():
    """Regression cases for the false-positive bug that fused 'DHAHANA PUTRA'
    (a signing official's name) with an unrelated trailing fragment. A plain
    long all-caps sentence must never be split just because it ends in a
    lowercase-adjacent token (number, comma, hyphenated word)."""
    cases = [
        "PERATURAN MENTERI KEUANGAN TENTANG PENGELOLAAN DANA BAGI HASIL DAN DANA ALOKASI UMUM.",
        "DIREKTUR JENDERAL PERATURAN PERUNDANG-UNDANGAN KEMENTERIAN HUKUM REPUBLIK INDONESIA,",
        "BERITA NEGARA REPUBLIK INDONESIA TAHUN 2026 NOMOR",
        "MENTERI KEUANGAN REPUBLIK INDONESIA,",
    ]
    for text in cases:
        heading, remainder = _split_embedded_heading(text)
        assert heading is None, f"incorrectly split: {text!r} -> heading={heading!r}"
        assert remainder == text


def test_embedded_heading_recovered_and_ordered_before_paragraph_it_precedes():
    """The recovered heading must be inserted at its true position in the
    source PDF (using bbox top), not wherever Docling happened to order the
    fused item. In the real PKSOP4D case the heading sat physically above
    the formula it got fused into — the merged item's bbox starts higher
    on the page (larger 'top' in BOTTOMLEFT coords) than the paragraph
    before it."""
    texts = [
        _text_item(0, "Nilai Kinerja = (0,1 x", label="text", top=110.0, bottom=100.0),
        _text_item(
            1,
            "RUMUS PERHITUNGAN NILAI KINERJA PKSOP4D) + (0,2 x Kepatuhan Pelaporan SPT)",
            label="text",
            top=130.0,
            bottom=90.0,
        ),
    ]
    raw = _raw_doc(texts=texts, body_refs=["#/texts/0", "#/texts/1"])

    md = _build_rich_markdown(raw)

    heading_pos = md.index("RUMUS PERHITUNGAN NILAI KINERJA")
    formula_pos = md.index("Nilai Kinerja = (0,1 x PKSOP4D)")
    assert heading_pos < formula_pos, "heading must render before the paragraph it precedes in the PDF"


def test_embedded_heading_no_reorder_when_not_physically_above():
    """When the fused item is NOT physically above the previous paragraph,
    the heading must stay appended after it (no spurious reordering)."""
    texts = [
        _text_item(0, "Nilai Kinerja = (0,1 x", label="text", top=130.0, bottom=120.0),
        _text_item(
            1,
            "RUMUS PERHITUNGAN NILAI KINERJA PKSOP4D) + (0,2 x Kepatuhan Pelaporan SPT)",
            label="text",
            top=110.0,
            bottom=70.0,
        ),
    ]
    raw = _raw_doc(texts=texts, body_refs=["#/texts/0", "#/texts/1"])

    md = _build_rich_markdown(raw)

    heading_pos = md.index("RUMUS PERHITUNGAN NILAI KINERJA")
    formula_pos = md.index("Nilai Kinerja = (0,1 x PKSOP4D)")
    assert formula_pos < heading_pos


# -----------------------------------------------------------------------
# Blank-line paragraph separation (CommonMark soft-break collapse)
# -----------------------------------------------------------------------

def test_list_items_separated_by_blank_lines():
    """Consecutive list_item lines need a blank line between them, or
    CommonMark renderers collapse them into a single paragraph (soft line
    breaks render as spaces in most viewers)."""
    texts = [
        _text_item(0, "first clause text", label="list_item", marker="a."),
        _text_item(1, "second clause text", label="list_item", marker="b."),
    ]
    groups = [_group_item(0, ["#/texts/0", "#/texts/1"])]
    raw = _raw_doc(texts=texts, groups=groups, body_refs=["#/groups/0"])

    md = _build_rich_markdown(raw)
    lines = md.splitlines()

    idx_b = next(i for i, l in enumerate(lines) if l.startswith("b."))
    assert lines[idx_b - 1] == "", "expected a blank line before the second list item"
