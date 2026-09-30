"""Engine adapter argument construction and log parsing.

The log excerpts here are captured verbatim from vLLM v0.26.0 running
Llama-3.1-8B on the target A40, so these tests pin behaviour against what the
engine actually emits rather than what documentation claims.

Two of these tests exist because the real launch failed first:

* ``--model`` is deprecated in v0.26; the model is a positional argument.
* ``--disable-log-requests`` was **removed** in v0.26 and replaced by the
  inverted ``--enable-log-requests`` / ``--no-enable-log-requests``.

Both produced an immediate container exit. Catching a recurrence in CI is far
cheaper than catching it at the start of a lab session.
"""

from __future__ import annotations

from pathlib import Path

from llmbench.config import load_engine_profile
from llmbench.engines.base import EngineLaunchSpec
from llmbench.engines.sglang import SglangEngine
from llmbench.engines.vllm import VllmEngine
from llmbench.schema import EngineName

# Verbatim from a real v0.26.0 startup, including the ASCII-art banner line.
VLLM_BANNER_LOG = """\
(APIServer pid=1) INFO 08-10 06:25:29 [api_utils.py:345]        █     █     █▄   ▄█
(APIServer pid=1) INFO 08-10 06:25:29 [api_utils.py:345]  ▄▄ ██ █     █     █ ▀▄▀ █  version 0.26.0
(EngineCore pid=86) INFO 08-10 06:25:58 [core.py:116] Initializing a V1 LLM engine (v0.26.0) with config: model='/root/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct'
(EngineCore pid=86) INFO 08-10 06:28:08 [gpu_worker.py:560] Available KV cache memory: 23.9 GiB
(EngineCore pid=86) INFO 08-10 06:28:08 [kv_cache_utils.py:2177] GPU KV cache size: 195,760 tokens
(EngineCore pid=86) INFO 08-10 06:28:29 [core.py:340] init engine (profile, create kv cache, warmup model) took 54.24 s (compilation: 29.60 s)
"""


def spec(**overrides: object) -> EngineLaunchSpec:
    base: dict[str, object] = {
        "config_id": "vllm-bf16",
        "image": "vllm/vllm-openai",
        "tag": "v0.26.0",
        "image_digest": "sha256:" + "ff" * 32,
        "model_hf_id": "meta-llama/Llama-3.1-8B-Instruct",
        "model_revision": "0e9e39f249a16976918f6564b8830bc894c89659",
        "gpu_index": 1,
        "port": 8000,
        "max_model_len": 4096,
        "gpu_memory_utilization": 0.90,
        "max_num_seqs": 256,
        "hf_cache_dir": Path("/home/ubuntu/.cache/huggingface"),
        "expected_kernel": None,
        "kernel_log_patterns": {},
    }
    base.update(overrides)
    return EngineLaunchSpec(**base)  # type: ignore[arg-type]


def profile_spec(engine: EngineName, **overrides: object) -> EngineLaunchSpec:
    """A spec carrying the committed profile's shared flags, as the runners build it."""
    return spec(extra_args=dict(load_engine_profile(engine).args), **overrides)


class TestVllmArguments:
    def test_model_is_positional_not_a_flag(self) -> None:
        """v0.26 deprecated `--model`; a real launch warned and then failed."""
        args = VllmEngine().server_args(spec())
        assert args[0] == "meta-llama/Llama-3.1-8B-Instruct"
        assert "--model" not in args

    def test_uses_the_inverted_log_requests_flag(self) -> None:
        """`--disable-log-requests` was removed in v0.26.

        Passing it caused `error: unrecognized arguments` and an immediate
        container exit.
        """
        args = VllmEngine().server_args(profile_spec(EngineName.VLLM))
        assert "--no-enable-log-requests" in args
        assert "--disable-log-requests" not in args

    def test_pins_revision_not_a_branch(self) -> None:
        args = VllmEngine().server_args(spec())
        assert "--revision" in args
        assert args[args.index("--revision") + 1] == "0e9e39f249a16976918f6564b8830bc894c89659"

    def test_carries_the_parity_critical_settings(self) -> None:
        """These must match across engines or the comparison is meaningless."""
        args = VllmEngine().server_args(profile_spec(EngineName.VLLM))
        for flag, value in [
            ("--max-model-len", "4096"),
            ("--gpu-memory-utilization", "0.9"),
            ("--max-num-seqs", "256"),
            ("--seed", "0"),
        ]:
            assert flag in args
            assert args[args.index(flag) + 1] == value


class TestVllmLogParsing:
    def test_parses_version_from_the_real_banner(self) -> None:
        assert VllmEngine().parse_version(VLLM_BANNER_LOG) == "0.26.0"

    def test_parses_version_from_the_engine_core_line_alone(self) -> None:
        """Belt and braces: a cosmetic banner change must not degrade the
        recorded version to 'unknown'."""
        line = "INFO [core.py:116] Initializing a V1 LLM engine (v0.26.0) with config: model='x'"
        assert VllmEngine().parse_version(line) == "0.26.0"

    def test_unknown_version_does_not_raise(self) -> None:
        assert VllmEngine().parse_version("no version here") == "unknown"


class TestDockerArguments:
    def test_runs_against_a_digest_never_a_tag(self) -> None:
        """A moving tag makes every earlier result unreproducible."""
        args = VllmEngine().docker_args(spec())
        image_ref = args[-1]
        assert "@sha256:" in image_ref
        assert ":v0.26.0" not in image_ref

    def test_selects_the_requested_host_gpu(self) -> None:
        args = VllmEngine().docker_args(spec(gpu_index=1))
        assert '"device=1"' in args

    def test_mounts_the_hf_cache_read_only(self) -> None:
        """Reusing the host cache avoids re-downloading 15 GB per image, and
        read-only guarantees the benchmark cannot mutate what it measures."""
        args = VllmEngine().docker_args(spec())
        assert any(a.endswith(":/root/.cache/huggingface:ro") for a in args)

    def test_engine_never_needs_the_hub(self) -> None:
        """Weights are pre-cached and mounted read-only; a Hub lookup can only
        cost start-up time, and with failing DNS it cost the whole deadline."""
        args = VllmEngine().docker_args(spec())
        assert args[args.index("-e") + 1] == "HF_HUB_OFFLINE=1"

    def test_raises_shm_above_the_docker_default(self) -> None:
        """Docker's default 64 MB /dev/shm is too small for vLLM worker IPC."""
        args = VllmEngine().docker_args(spec())
        assert "--shm-size" in args
        assert args[args.index("--shm-size") + 1] == "16g"


class TestSglangArguments:
    def test_maps_parity_settings_onto_sglang_names(self) -> None:
        """The knobs are not identically named, and one is not semantically
        identical either — documented rather than glossed over."""
        args = SglangEngine().server_args(spec(config_id="sglang-bf16"))
        assert "--context-length" in args  # vLLM's --max-model-len
        assert "--mem-fraction-static" in args  # vLLM's --gpu-memory-utilization
        assert "--max-running-requests" in args  # vLLM's --max-num-seqs

    def test_enables_metrics_explicitly(self) -> None:
        """Prometheus metrics are opt-in on SGLang, unlike vLLM."""
        args = SglangEngine().server_args(profile_spec(EngineName.SGLANG, config_id="sglang-bf16"))
        assert "--enable-metrics" in args


class TestProfileArguments:
    """The shared flags come from configs/engines/*.yaml, not from code.

    They used to be hard-coded in the adapters while the YAML listed the same
    flags and was never read, so editing the file changed nothing. Moving them
    must not change what the engines are launched with: the full command line
    is pinned here, exactly as every committed result was produced.
    """

    def test_vllm_command_line_is_unchanged(self) -> None:
        assert VllmEngine().server_args(profile_spec(EngineName.VLLM)) == [
            "meta-llama/Llama-3.1-8B-Instruct",
            "--revision",
            "0e9e39f249a16976918f6564b8830bc894c89659",
            "--port",
            "8000",
            "--host",
            "0.0.0.0",
            "--max-model-len",
            "4096",
            "--gpu-memory-utilization",
            "0.9",
            "--max-num-seqs",
            "256",
            "--no-enable-log-requests",
            "--seed",
            "0",
        ]

    def test_sglang_command_line_is_unchanged(self) -> None:
        args = SglangEngine().server_args(profile_spec(EngineName.SGLANG, config_id="sglang-bf16"))
        assert args[args.index("--max-running-requests") :] == [
            "--max-running-requests",
            "256",
            "--enable-metrics",
            "--random-seed",
            "0",
        ]

    def test_without_the_profile_nothing_is_added(self) -> None:
        """No hidden defaults: an adapter adds only what the spec carries."""
        args = VllmEngine().server_args(spec())
        assert "--seed" not in args
        assert "--no-enable-log-requests" not in args


class TestVersionFallback:
    def test_version_comes_from_the_tag_when_the_log_has_none(self) -> None:
        """SGLang prints no version at start-up, which left 48 records saying
        'unknown'. The tag names the release; the digest pins the build."""
        from llmbench.engines.base import _version_from_tag  # noqa: PLC0415

        assert _version_from_tag("v0.5.17-cu129") == "0.5.17"
        assert _version_from_tag("v0.26.0") == "0.26.0"
        assert _version_from_tag("latest") is None


class TestNetworkExposure:
    def test_engine_api_is_never_published_on_every_interface(self) -> None:
        """The OpenAI API has no authentication; on a shared machine it must not
        listen on every interface. By default it is published on loopback only."""
        args = VllmEngine().docker_args(spec())
        published = [args[i + 1] for i, a in enumerate(args) if a == "-p"]
        assert published == ["127.0.0.1:8000:8000"]

    def test_bridge_gateway_is_added_for_the_metrics_stack(self) -> None:
        """Prometheus scrapes host.docker.internal, i.e. the Docker bridge
        gateway -- local to the machine, unlike 0.0.0.0."""
        args = VllmEngine().docker_args(spec(publish_addresses=("127.0.0.1", "172.17.0.1")))
        published = [args[i + 1] for i, a in enumerate(args) if a == "-p"]
        assert published == ["127.0.0.1:8000:8000", "172.17.0.1:8000:8000"]
        assert not any(p.startswith("0.0.0.0") or p == "8000:8000" for p in published)
