"""The static results explorer.

Its logic is exercised under node elsewhere with the chart library stubbed —
which is precisely how a pinned CDN URL that returned 404 went unnoticed for
two releases while the slider and table kept working. These tests check the
things a stub cannot.
"""

from __future__ import annotations

import re
import urllib.error
import urllib.request
from pathlib import Path

import pytest

INDEX = Path(__file__).resolve().parent.parent / "docs" / "index.html"


def external_scripts() -> list[str]:
    return re.findall(r'<script\s+src="(https://[^"]+)"', INDEX.read_text())


class TestExternalDependencies:
    def test_exactly_one_external_script(self) -> None:
        """One dependency, pinned. More is more ways to break."""
        assert len(external_scripts()) == 1

    def test_script_is_version_pinned(self) -> None:
        (url,) = external_scripts()
        assert re.search(r"/\d+\.\d+\.\d+/", url), url

    def test_script_has_an_onerror_handler(self) -> None:
        """If the CDN fails, the page must say so rather than show an empty
        panel that looks like a page with no data."""
        assert "onerror=" in INDEX.read_text()
        assert 'typeof Plotly === "undefined"' in INDEX.read_text()

    def test_pinned_url_actually_exists(self) -> None:
        """A HEAD request against the real CDN. Skipped without network;
        a 404 is a failure, not a skip — it is exactly the regression."""
        (url,) = external_scripts()
        req = urllib.request.Request(url, method="HEAD")
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                assert resp.status == 200
        except urllib.error.HTTPError as exc:
            pytest.fail(f"{url} -> HTTP {exc.code}: the explorer's chart will not render")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            pytest.skip(f"no network: {exc}")


class TestDataContract:
    def test_results_json_is_fetched_relatively(self) -> None:
        """Absolute paths would break the moment the page is served from a
        different origin — GitHub Pages and a Hugging Face Space both host it."""
        assert 'fetch("results.json")' in INDEX.read_text()
