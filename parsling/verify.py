"""
parsling/verify.py
~~~~~~~~~~~~~~~~~~~
Two-tier verification for DoclingDocument JSON output. parsling is fully
offline-capable without this module's LLM tier — flag_pages() runs purely
on local JSON metadata, and verify_document() is an opt-in add-on for
when network access and an API key are available.

    1. flag_pages()     — free heuristic triage over the JSON metadata.
                           No network call, no LLM cost, fully offline.
                           Finds the same class of artifact
                           _build_rich_markdown's recovery heuristics
                           already patch (embedded headings, page-split
                           fragments, ...) plus a few signals that aren't
                           safe to silently auto-fix (empty table grids,
                           OCR-repeat garble).

    2. verify_document() — OPT-IN, requires network + an API key. For
                           only the pages flag_pages() actually flagged,
                           render the original PDF page to an image and
                           ask a vision-capable LLM to check the
                           extracted text against what's really on the
                           page. This is the expensive tier, so it never
                           runs on pages the free heuristics didn't flag,
                           and the CLI never invokes it unless --llm is
                           explicitly passed.

Step 2 talks to any OpenAI-compatible vision endpoint. Defaults to
OpenAI (gpt-5.4-nano — cheaper and more precise than gpt-4o-mini in
calibration testing on this project's documents) — note DeepSeek's chat
completions API does NOT accept image content despite some third-party
claims to the contrary (confirmed against the live API: every model name
returns the same "unknown variant `image_url`" deserialization error).
Any other OpenAI-compatible vision provider works via --base-url/--model,
as long as it actually accepts image_url content blocks.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from parsling.exporters import _dedupe_repeated_phrases, _split_embedded_heading

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-5.4-nano"
_API_KEY_ENV_VARS = ("OPENAI_API_KEY", "DEEPSEEK_API_KEY")

_DEDUPE_SHRINK_RATIO = 0.7    # flag if de-duplication removes more than 30% of words
_MIN_WORDS_FOR_DEDUPE_CHECK = 6


# ---------------------------------------------------------------------------
# Tier 1 — free heuristic triage
# ---------------------------------------------------------------------------

def flag_pages(raw: dict) -> dict[int, list[str]]:
    """
    Scan a raw DoclingDocument JSON dict for likely extraction artifacts.

    Pure metadata analysis — no network call, no LLM, effectively free.
    Returns ``{page_no: [reason, ...]}`` for pages worth a closer look.
    Pages with no flags are simply absent from the result.
    """
    flags: dict[int, list[str]] = defaultdict(list)

    for t in raw.get("texts", []):
        label = t.get("label")
        text = (t.get("text") or "").strip()
        page = _item_page(t)
        if page is None or not text:
            continue

        if label == "text":
            heading, _ = _split_embedded_heading(text)
            if heading:
                flags[page].append(
                    f"embedded heading fused into body text: {heading!r}"
                )

        if label in ("text", "list_item"):
            words = text.split()
            if len(words) >= _MIN_WORDS_FOR_DEDUPE_CHECK:
                deduped_words = _dedupe_repeated_phrases(text).split()
                if len(deduped_words) <= len(words) * _DEDUPE_SHRINK_RATIO:
                    flags[page].append(
                        "text shrank >30% after de-duplication "
                        "(possible OCR repeat/garble)"
                    )

        if label == "list_item" and not (t.get("marker") or "").strip():
            flags[page].append(
                "unmarked list item (possible page-split continuation fragment)"
            )

    for tb in raw.get("tables", []):
        page = _item_page(tb)
        grid = (tb.get("data") or {}).get("grid") or []
        if page is not None and not grid:
            flags[page].append("table has no grid data")

    return dict(flags)


def _item_page(item: dict) -> int | None:
    prov = item.get("prov") or []
    return prov[0].get("page_no") if prov else None


# ---------------------------------------------------------------------------
# Per-page extracted text (context sent to the verifier model)
# ---------------------------------------------------------------------------

def _build_page_texts(raw: dict) -> dict[int, list[str]]:
    """
    Build per-page text content directly from each item's own provenance
    (``prov[0].page_no``), independent of body-tree position or rendered
    Markdown layout.

    Earlier this scraped page numbers back out of the rendered Markdown's
    ``<!-- ... page=N ... -->`` comments, which silently dropped content
    near page boundaries (list items have no trailing comment to attach
    to, and merged tables emit a "83-84" range the page= regex only half
    matched) — pages came back with zero lines of context even though
    they were full of text, producing false "nothing was extracted"
    findings from the verifier model. Reading provenance directly is
    immune to that class of bug.
    """
    pages: dict[int, list[str]] = defaultdict(list)

    for t in raw.get("texts", []):
        text = (t.get("text") or "").strip()
        prov = t.get("prov") or []
        if not text or not prov:
            continue
        # Docling stores a list_item's enumeration marker ("a.", "(12)", ...)
        # separately from its text, the same way the rich-markdown exporter
        # does. Omitting it here (as an earlier version of this function
        # did) made the verifier "discover" every marker as missing —
        # a false positive in this prompt-context builder, not a real
        # extraction gap.
        marker = (t.get("marker") or "").strip() if t.get("label") == "list_item" else ""
        line = f"{marker} {text}".strip() if marker else text
        page = prov[0].get("page_no")
        if page is not None:
            pages[page].append(line)

    for tb in raw.get("tables", []):
        grid = (tb.get("data") or {}).get("grid") or []
        prov = tb.get("prov") or []
        if not grid or not prov:
            continue
        page = prov[0].get("page_no")
        if page is None:
            continue
        rows = [" | ".join((c.get("text") or "").strip() for c in row) for row in grid]
        pages[page].append("[TABLE] " + " ; ".join(rows))

    return dict(pages)


# ---------------------------------------------------------------------------
# Tier 2 — vision LLM verification
# ---------------------------------------------------------------------------

_VERIFY_PROMPT = """You are proofreading a PDF text-extraction pipeline's output against the \
original page image.

Extracted text for this page:
---
{extracted_text}
---

Compare it against the page image. Report ONLY genuine content discrepancies: missing \
words/sentences, garbled or duplicated text, numbers/dates/names that don't match, or content \
that's clearly out of order. Ignore formatting differences (markdown syntax, line breaks, bullet \
style, heading levels) — those are expected and not errors.

It is normal and expected for most pages to have ZERO issues — do not invent one to fill the \
list. Before reporting any issue, re-read its "extracted_says" and "page_actually_says" values: \
if they would end up saying the same thing, that is NOT an issue — drop it. Only include an \
issue where a careful reader would see the extracted text say something different from, or less \
than, what the image actually shows.

Respond with strict JSON only, no prose outside the JSON, in this exact shape:
{{"status": "ok" or "issues_found", "issues": [{{"description": "...", "extracted_says": "...", \
"page_actually_says": "..."}}]}}
If status is "ok", "issues" must be an empty list.
"""


@dataclass
class PageVerification:
    page_no: int
    flags: list[str]
    status: str  # "ok" | "issues_found" | "error"
    issues: list[dict] = field(default_factory=list)
    raw_response: str | None = None


class VisionVerifier:
    """Thin client for an OpenAI-compatible vision chat endpoint."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 120.0,
    ) -> None:
        import httpx

        if api_key is None:
            for var in _API_KEY_ENV_VARS:
                api_key = os.environ.get(var)
                if api_key:
                    break
        self.api_key = api_key
        if not self.api_key:
            raise ValueError(
                "No API key found. Pass api_key=, or set one of "
                f"{_API_KEY_ENV_VARS} (e.g. in a .env file — parsling already loads it)."
            )
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout)

    def verify_page(self, image_png: bytes, extracted_text: str) -> dict:
        b64 = base64.b64encode(image_png).decode("ascii")
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": _VERIFY_PROMPT.format(
                                extracted_text=extracted_text or "(nothing extracted for this page)"
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                    ],
                }
            ],
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.base_url}/chat/completions"

        max_attempts = 4
        for attempt in range(1, max_attempts + 1):
            resp = self._client.post(url, headers=headers, json=payload)
            if resp.status_code == 429 and attempt < max_attempts:
                wait = float(resp.headers.get("retry-after", 2 ** attempt))
                logger.info("Rate limited, retrying in %.0fs (attempt %d/%d)", wait, attempt, max_attempts)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            return _drop_noop_issues(_parse_json_response(content))
        resp.raise_for_status()  # pragma: no cover — exhausted retries, surface last error
        return {}

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "VisionVerifier":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _parse_json_response(content: str) -> dict:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Verifier returned non-JSON content; treating as error: %.200s", text)
        return {"status": "error", "issues": [], "_raw": content}


_WS_RE = re.compile(r"\s+")


def _drop_noop_issues(parsed: dict) -> dict:
    """
    Belt-and-suspenders against hallucinated "issues": even though the
    prompt now tells the model not to report a discrepancy where
    extracted_says == page_actually_says, cheap models don't always
    follow that instruction (observed in practice: gpt-4o-mini reported
    a 17-item "issues_found" page where every single pair was character-
    for-character identical). Filter those out in code rather than
    trusting the model's own "status" judgment.
    """
    issues = parsed.get("issues")
    if not isinstance(issues, list):
        return parsed
    kept = [
        i
        for i in issues
        if not (
            isinstance(i, dict)
            and _WS_RE.sub(" ", str(i.get("extracted_says", "")).strip())
            == _WS_RE.sub(" ", str(i.get("page_actually_says", "")).strip())
        )
    ]
    parsed["issues"] = kept
    if not kept and parsed.get("status") == "issues_found":
        parsed["status"] = "ok"
    return parsed


def render_page_png(pdf_path: str | Path, page_no: int, scale: float = 2.0) -> bytes:
    """Render a single 1-indexed PDF page to PNG bytes."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        page = pdf[page_no - 1]
        bitmap = page.render(scale=scale)
        buf = io.BytesIO()
        bitmap.to_pil().save(buf, format="PNG")
        return buf.getvalue()
    finally:
        pdf.close()


def verify_document(
    raw: dict,
    pdf_path: str | Path,
    flags: dict[int, list[str]] | None = None,
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    max_pages: int | None = None,
    image_scale: float = 2.0,
) -> list[PageVerification]:
    """
    Run the vision-LLM verification pass on flagged pages only.

    Parameters
    ----------
    raw:
        The raw DoclingDocument JSON dict (``json.loads`` of the saved file).
    pdf_path:
        Path to the original source PDF — needed to render page images.
    flags:
        Pre-computed ``flag_pages(raw)`` result. Computed automatically if
        omitted.
    max_pages:
        Cap how many flagged pages get sent to the model, for cost control.
        Pages are checked in ascending page-number order.

    Returns
    -------
    list[PageVerification]
        One result per checked page, in ascending page-number order.
    """
    if flags is None:
        flags = flag_pages(raw)
    if not flags:
        return []

    pages_to_check = sorted(flags)
    if max_pages is not None:
        pages_to_check = pages_to_check[:max_pages]

    page_texts = _build_page_texts(raw)

    results: list[PageVerification] = []
    with VisionVerifier(api_key=api_key, model=model, base_url=base_url) as verifier:
        for page_no in pages_to_check:
            try:
                image = render_page_png(pdf_path, page_no, scale=image_scale)
                text = "\n".join(page_texts.get(page_no, []))
                parsed = verifier.verify_page(image, text)
                results.append(
                    PageVerification(
                        page_no=page_no,
                        flags=flags[page_no],
                        status=parsed.get("status", "error"),
                        issues=parsed.get("issues", []),
                        raw_response=parsed.get("_raw"),
                    )
                )
            except Exception as exc:
                logger.warning("Verification failed for page %d: %s", page_no, exc)
                results.append(
                    PageVerification(
                        page_no=page_no,
                        flags=flags[page_no],
                        status="error",
                        issues=[],
                        raw_response=str(exc),
                    )
                )
    return results
