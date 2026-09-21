"""Publish the results explorer to a Hugging Face static Space.

A *static* Space, deliberately: it serves the same ``index.html`` and
``results.json`` that GitHub Pages does, with no Gradio runtime to fall asleep,
no Python environment to drift, and nothing that can be out of date relative to
``results/`` except by not having been redeployed. ``make report`` regenerates
``docs/``; this script ships it.

Source of truth stays in the repository. The Space is assembled from ``docs/``
and ``results/figures/`` at deploy time and nothing is hand-maintained on the
Hub side.

Usage:
    HF_TOKEN=hf_... uv run python scripts/deploy_hf_space.py --space <user>/<name>
    uv run python scripts/deploy_hf_space.py --space <user>/<name> --dry-run
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GITHUB = "https://github.com/vireshkoli/LLM-Inference-Optimization"
PAGES = "https://vireshkoli.github.io/LLM-Inference-Optimization/"


def space_readme(space_id: str) -> str:
    """Space card: frontmatter the Hub requires, then a real summary.

    The summary is the repository README's generated headline block, copied
    verbatim so the card cannot say something the results do not.
    """
    readme = (REPO / "README.md").read_text()
    start = readme.index("<!-- BEGIN:headline -->") + len("<!-- BEGIN:headline -->\n")
    end = readme.index("\n<!-- END:headline -->")
    headline = readme[start:end].strip()
    return f"""---
title: LLM Inference Optimization
emoji: 📊
colorFrom: blue
colorTo: green
sdk: static
pinned: false
license: mit
short_description: Quantized Llama-3.1-8B serving on an A40, measured open-loop
---

# LLM Inference Optimization — quantization on an NVIDIA A40

Latency-vs-throughput curves, quality, and cost for quantized Llama-3.1-8B serving on a
bandwidth-bound GPU — measured open-loop, with the tail reported and the measurement's own
validity checked. **The methodology is the product; the numbers are its output.**

{headline}

![Quality vs cost at a fixed latency budget](figures/pareto_quality_cost.png)

**The interactive explorer is the app above**: set a p95 TTFT budget and see which
configuration is cheapest under it.

- **Repository, methodology and full report:** {GITHUB}
- **Findings:** {GITHUB}/blob/main/REPORT.md
- **How it was measured, and which numbers are not trustworthy:** {GITHUB}/blob/main/METHODOLOGY.md

Every number here is generated from `results/*.json` in the repository by `make report`;
nothing is hand-typed. This Space is `{space_id}`, deployed by `make deploy-hf`.
"""


def assemble(space_id: str, out: Path) -> list[Path]:
    """Lay out the Space's files from the repository's generated artifacts."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)

    copied: list[Path] = []
    for src, dst in (
        (REPO / "docs" / "index.html", out / "index.html"),
        (REPO / "docs" / "results.json", out / "results.json"),
    ):
        if not src.exists():
            sys.exit(f"missing {src}; run `make report` first")
        shutil.copy2(src, dst)
        copied.append(dst)
    for png in sorted((REPO / "results" / "figures").glob("*.png")):
        shutil.copy2(png, out / "figures" / png.name)
        copied.append(out / "figures" / png.name)

    (out / "README.md").write_text(space_readme(space_id))
    copied.append(out / "README.md")
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--space", required=True, help="<user>/<name>")
    parser.add_argument("--dry-run", action="store_true", help="assemble only; do not push")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp) / "space"
        files = assemble(args.space, staged)
        print(f"assembled {len(files)} file(s) for {args.space}:")
        for f in files:
            print(f"  {f.relative_to(staged)}  ({f.stat().st_size:,} bytes)")
        if args.dry_run:
            return

        token = os.environ.get("HF_TOKEN")
        if not token:
            sys.exit(
                "HF_TOKEN is not set. This machine is shared and the cached Hub login may not "
                "be yours; pass your own write token explicitly rather than relying on it."
            )

        from huggingface_hub import HfApi  # noqa: PLC0415

        api = HfApi(token=token)
        me = api.whoami()
        owner = args.space.split("/")[0]
        if me.get("name") != owner and owner not in {o.get("name") for o in me.get("orgs", [])}:
            sys.exit(f"token belongs to {me.get('name')!r}, not to the Space owner {owner!r}")

        api.create_repo(args.space, repo_type="space", space_sdk="static", exist_ok=True)
        api.upload_folder(
            folder_path=str(staged),
            repo_id=args.space,
            repo_type="space",
            commit_message="deploy: results explorer from `make report`",
        )
        print(f"deployed: https://huggingface.co/spaces/{args.space}")


if __name__ == "__main__":
    main()
