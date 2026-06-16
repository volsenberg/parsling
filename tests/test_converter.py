"""
tests/test_converter.py
~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for PdfParser — uses a small test PDF.
Run with: pytest tests/ -v
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from parsling.converter import PdfParser, ParseResult, _resolve_profile
from parsling.config import ParseProfile
from parsling.profiles import FAST, ACCURATE, VLM
from docling.datamodel.base_models import ConversionStatus


# -----------------------------------------------------------------------
# Profile resolution
# -----------------------------------------------------------------------

def test_resolve_profile_by_string():
    assert _resolve_profile("fast") is FAST
    assert _resolve_profile("ACCURATE") is ACCURATE
    assert _resolve_profile("vlm") is VLM


def test_resolve_profile_by_instance():
    custom = ParseProfile(use_vlm=False, ocr_engine="tesseract_cli", ocr_langs=["eng"])
    assert _resolve_profile(custom) is custom


def test_resolve_profile_unknown():
    with pytest.raises(ValueError, match="Unknown profile"):
        _resolve_profile("ultrafast")


# -----------------------------------------------------------------------
# PdfParser construction
# -----------------------------------------------------------------------

def test_parser_default_profile():
    with patch("parsling.converter._build_standard_converter") as mock_build:
        mock_build.return_value = MagicMock()
        parser = PdfParser()
    assert parser.profile is ACCURATE


def test_parser_fast_profile():
    with patch("parsling.converter._build_standard_converter") as mock_build:
        mock_build.return_value = MagicMock()
        parser = PdfParser(profile="fast")
    assert parser.profile is FAST
    assert not parser.profile.use_vlm


def test_parser_vlm_profile():
    with patch("parsling.converter._build_vlm_converter") as mock_build:
        mock_build.return_value = MagicMock()
        parser = PdfParser(profile="vlm")
    assert parser.profile is VLM
    assert parser.profile.use_vlm


# -----------------------------------------------------------------------
# ParseResult helpers
# -----------------------------------------------------------------------

def test_parse_result_ok_on_success():
    r = ParseResult(document=MagicMock(), source=Path("x.pdf"), status=ConversionStatus.SUCCESS)
    assert r.ok is True


def test_parse_result_ok_on_partial():
    r = ParseResult(document=MagicMock(), source=Path("x.pdf"), status=ConversionStatus.PARTIAL_SUCCESS)
    assert r.ok is True


def test_parse_result_not_ok_on_failure():
    r = ParseResult(document=None, source=Path("x.pdf"), status=ConversionStatus.FAILURE, error="boom")
    assert r.ok is False


# -----------------------------------------------------------------------
# parse() — mocked converter
# -----------------------------------------------------------------------

def _make_mock_result(status=ConversionStatus.SUCCESS):
    doc = MagicMock()
    conv_result = MagicMock()
    conv_result.status = status
    conv_result.document = doc
    return conv_result, doc


def test_parse_returns_document(tmp_path):
    pdf = tmp_path / "test.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    conv_result, doc = _make_mock_result()

    with patch("parsling.converter._build_standard_converter") as mock_build:
        mock_converter = MagicMock()
        mock_converter.convert.return_value = conv_result
        mock_build.return_value = mock_converter

        parser = PdfParser(profile="fast")
        result = parser.parse(pdf)

    assert result is doc


def test_parse_raises_on_failure(tmp_path):
    pdf = tmp_path / "bad.pdf"
    pdf.write_bytes(b"not a pdf")

    conv_result, _ = _make_mock_result(status=ConversionStatus.FAILURE)

    with patch("parsling.converter._build_standard_converter") as mock_build:
        mock_converter = MagicMock()
        mock_converter.convert.return_value = conv_result
        mock_build.return_value = mock_converter

        parser = PdfParser(profile="fast")
        with pytest.raises(RuntimeError):
            parser.parse(pdf)


# -----------------------------------------------------------------------
# parse_folder()
# -----------------------------------------------------------------------

def test_parse_folder_no_pdfs(tmp_path):
    with patch("parsling.converter._build_standard_converter") as mock_build:
        mock_build.return_value = MagicMock()
        parser = PdfParser(profile="fast")

    results = list(parser.parse_folder(tmp_path))
    assert results == []
