"""vLLM engine adapter.

vLLM is the primary engine: it carries the full quantization axis, has the best
Ampere kernel coverage (gptq_marlin, awq_marlin, compressed-tensors INT8), and
exposes the Prometheus histograms the observability stack needs.
"""

from __future__ import annotations

import re

from llmbench.engines.base import EngineLaunchSpec, EngineProcess

__all__ = ["VllmEngine"]

# Verified against real v0.26.0 output. The banner line is ASCII art with the
# version appended, and the engine-core line carries it in parentheses; both are
# matched so a cosmetic banner change cannot silently degrade the record to
# "unknown".
_VERSION_PATTERNS = (
    r"Initializing a V\d+ LLM engine \(v([0-9][^\s)]*)\)",
    r"\bversion\s+([0-9]+\.[0-9]+\.[0-9]+[^\s,)]*)",
    r"vLLM API server version ([0-9][^\s,)]*)",
)


class VllmEngine(EngineProcess):
    name = "vllm"

    def server_args(self, spec: EngineLaunchSpec) -> list[str]:
        args = [
            # Positional since v0.26; `--model` is deprecated and warns.
            spec.model_hf_id,
            "--revision",
            spec.model_revision,
            "--port",
            str(spec.port),
            "--host",
            "0.0.0.0",
            # Held identical across engines and quantization levels; the whole
            # cross-configuration comparison depends on these matching.
            "--max-model-len",
            str(spec.max_model_len),
            "--gpu-memory-utilization",
            str(spec.gpu_memory_utilization),
            "--max-num-seqs",
            str(spec.max_num_seqs),
            # Per-request logging perturbs the measurement it is logging. The
            # old `--disable-log-requests` was removed in v0.26; the flag is now
            # inverted and defaults off, but stated explicitly so a future
            # default change cannot silently start logging mid-project.
            "--no-enable-log-requests",
            "--seed",
            "0",
        ]
        for key, value in spec.extra_args.items():
            args.append(key)
            if value:
                args.append(value)
        return args

    def parse_version(self, startup_log: str) -> str:
        for pattern in _VERSION_PATTERNS:
            match = re.search(pattern, startup_log)
            if match:
                return match.group(1)
        return "unknown"
