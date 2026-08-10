"""SGLang engine adapter.

The secondary engine, running a reduced confirmation sweep. Its purpose is to
show that the quantization findings are not an artifact of vLLM specifically.

Held at parity with vLLM where the knobs correspond. Where they do not, the
asymmetry is named here and in METHODOLOGY.md rather than smoothed over:
``--mem-fraction-static`` is not semantically identical to vLLM's
``--gpu-memory-utilization``, and presenting them as equivalent would overstate
how controlled the comparison is.
"""

from __future__ import annotations

import re

from llmbench.engines.base import EngineLaunchSpec, EngineProcess

__all__ = ["SglangEngine"]

#: Documented in every result so the reader can judge the comparison.
PARITY_CAVEATS = {
    "gpu_memory_utilization": "maps to --mem-fraction-static (not semantically identical)",
    "max_num_seqs": "maps to --max-running-requests",
    "max_model_len": "maps to --context-length",
}


class SglangEngine(EngineProcess):
    name = "sglang"

    def server_args(self, spec: EngineLaunchSpec) -> list[str]:
        args = [
            "python3",
            "-m",
            "sglang.launch_server",
            "--model-path",
            spec.model_hf_id,
            "--revision",
            spec.model_revision,
            "--port",
            str(spec.port),
            "--host",
            "0.0.0.0",
            "--context-length",
            str(spec.max_model_len),
            "--mem-fraction-static",
            str(spec.gpu_memory_utilization),
            "--max-running-requests",
            str(spec.max_num_seqs),
            # Prometheus metrics are opt-in on SGLang, unlike vLLM.
            "--enable-metrics",
            "--random-seed",
            "0",
        ]
        for key, value in spec.extra_args.items():
            args.append(key)
            if value:
                args.append(value)
        return args

    def parse_version(self, startup_log: str) -> str:
        match = re.search(r"SGLang.*?version[= ]([0-9][^\s,)]*)", startup_log, re.IGNORECASE)
        return match.group(1) if match else "unknown"
