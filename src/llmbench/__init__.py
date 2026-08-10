"""llmbench — rigorous latency, throughput, quality and cost benchmarking of
quantized LLM serving.

The measurable claim this package exists to support is not "N tokens/sec" but a
latency-vs-throughput curve per configuration, swept to saturation, with the
tail reported and the measurement's own validity checked.
"""

from llmbench.schema import SCHEMA_VERSION

__all__ = ["SCHEMA_VERSION", "__version__"]

__version__ = "0.1.0"
