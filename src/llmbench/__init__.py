"""llmbench — rigorous latency, throughput, quality and cost benchmarking of
quantized LLM serving.

The measurable claim this package exists to support is not "N tokens/sec" but a
latency-vs-throughput curve per configuration, swept to saturation, with the
tail reported and the measurement's own validity checked.
"""

from importlib.metadata import PackageNotFoundError, version

from llmbench.schema import SCHEMA_VERSION

__all__ = ["SCHEMA_VERSION", "__version__"]

# One source of truth: pyproject.toml. A second, hand-kept copy here said 0.1.0
# through three releases.
try:
    __version__ = version("llmbench")
except PackageNotFoundError:  # a source tree that was never installed
    __version__ = "0.0.0+unknown"
