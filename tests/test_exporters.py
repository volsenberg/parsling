"""
tests/test_exporters.py
~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for DocExporter — mocks DoclingDocument to avoid real Docling calls.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

from parsling.exporters import DocExporter


def _make_doc(
    markdown="# Title\n\nBody text.",
    json_str=None,
    html="<html><body>doc</body></html>",
    doctags="<doc>mock</doc>",
    tables=None,
    pictures=None,
):
    """Build a minimal mock DoclingDocument."""
    doc = MagicMock()
    doc.export_to_markdown.return_value = markdown
    doc.model_dump_json.return_value = json_str or json.dumps({"mock": True}, indent=2)
    doc.model_dump.return_value = {"mock": True}
    doc.export_to_html.return_value = html
    doc.export_to_doctags.return_value = doctags

    mock_tables = tables or []
    doc.tables = mock_tables

    mock_pictures = pictures or []
    doc.pictures.__iter__ = MagicMock(return_value=iter(mock_pictures))

    try:
        doc.origin.filename = "test.pdf"
    except Exception:
        pass

    return doc


# -----------------------------------------------------------------------
# Text exports
# -----------------------------------------------------------------------

def test_to_markdown():
    doc = _make_doc(markdown="# Hello")
    exp = DocExporter(doc)
    assert exp.to_markdown() == "# Hello"


def test_to_json_valid():
    doc = _make_doc()
    exp = DocExporter(doc)
    data = json.loads(exp.to_json())
    assert data == {"mock": True}


def test_to_dict():
    doc = _make_doc()
    exp = DocExporter(doc)
    assert exp.to_dict() == {"mock": True}


def test_to_html_contains_body():
    doc = _make_doc()
    exp = DocExporter(doc)
    assert "<body>" in exp.to_html()


def test_to_doctags():
    doc = _make_doc(doctags="<doc>doctags</doc>")
    exp = DocExporter(doc)
    assert exp.to_doctags() == "<doc>doctags</doc>"


# -----------------------------------------------------------------------
# Tables
# -----------------------------------------------------------------------

def test_to_tables_empty():
    doc = _make_doc(tables=[])
    assert DocExporter(doc).to_tables() == []


def test_to_tables_returns_dataframes():
    import pandas as pd

    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    mock_table = MagicMock()
    mock_table.export_to_dataframe.return_value = df

    doc = _make_doc(tables=[mock_table])
    result = DocExporter(doc).to_tables()
    assert len(result) == 1
    assert list(result[0].columns) == ["A", "B"]


# -----------------------------------------------------------------------
# save() — filesystem
# -----------------------------------------------------------------------

def test_save_creates_markdown(tmp_path):
    doc = _make_doc(markdown="# Report")
    doc.tables = []
    doc.pictures.__iter__ = MagicMock(return_value=iter([]))
    doc.origin.filename = "test.pdf"

    exp = DocExporter(doc)
    exp.save(output_dir=tmp_path, stem="test", formats=["md"], save_figures=False)

    md_file = tmp_path / "test.md"
    assert md_file.exists()
    assert md_file.read_text() == "# Report"


def test_save_creates_json(tmp_path):
    doc = _make_doc()
    doc.tables = []
    doc.pictures.__iter__ = MagicMock(return_value=iter([]))
    # save_as_json is a file-writing method — mock it to simulate write
    doc.save_as_json = MagicMock(side_effect=lambda p, **kw: p.write_text('{"mock":true}'))

    DocExporter(doc).save(
        output_dir=tmp_path, stem="report", formats=["json"], save_figures=False
    )
    assert (tmp_path / "report.json").exists()


def test_save_creates_csv_per_table(tmp_path):
    import pandas as pd

    df = pd.DataFrame({"Col1": ["A", "B"], "Col2": [1, 2]})
    mock_table = MagicMock()
    mock_table.export_to_dataframe.return_value = df

    doc = _make_doc(tables=[mock_table])
    doc.pictures.__iter__ = MagicMock(return_value=iter([]))

    DocExporter(doc).save(
        output_dir=tmp_path, stem="fin", formats=["csv"], save_figures=False
    )
    csv_file = tmp_path / "tables" / "fin_table_01.csv"
    assert csv_file.exists()
    loaded = pd.read_csv(csv_file)
    assert list(loaded.columns) == ["Col1", "Col2"]


def test_save_html(tmp_path):
    doc = _make_doc(html="<html><body>test</body></html>")
    doc.tables = []
    doc.pictures.__iter__ = MagicMock(return_value=iter([]))

    DocExporter(doc).save(
        output_dir=tmp_path, stem="rep", formats=["html"], save_figures=False
    )
    html_file = tmp_path / "rep_html.md"
    assert html_file.exists()
    assert html_file.read_text() == "```html\n<html><body>test</body></html>\n```"


def test_save_creates_output_dir(tmp_path):
    doc = _make_doc()
    doc.tables = []
    doc.pictures.__iter__ = MagicMock(return_value=iter([]))

    new_dir = tmp_path / "deep" / "nested"
    DocExporter(doc).save(
        output_dir=new_dir, stem="x", formats=["md"], save_figures=False
    )
    assert new_dir.is_dir()


def test_save_creates_doctags(tmp_path):
    doc = _make_doc(doctags="<doc>doctags-save</doc>")
    doc.tables = []
    doc.pictures.__iter__ = MagicMock(return_value=iter([]))

    DocExporter(doc).save(
        output_dir=tmp_path, stem="rep", formats=["doctags"], save_figures=False
    )
    doctags_file = tmp_path / "rep_doctags.md"
    assert doctags_file.exists()
    assert doctags_file.read_text() == "```xml\n<doc>doctags-save</doc>\n```"
