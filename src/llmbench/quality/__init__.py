"""Quality evaluation — the third axis.

Quantization trades quality for speed, so a latency and cost benchmark that
ignores accuracy measures half the tradeoff. Everything here runs **through the
live serving engine** rather than through a separate transformers path, so the
quantized kernels actually under test are the ones being scored. A perplexity
computed on the checkpoint alone would miss a kernel bug entirely.
"""
