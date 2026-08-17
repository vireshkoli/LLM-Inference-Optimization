"""Engine container lifecycle.

Engines run as containers rather than in local virtualenvs. They carry
conflicting torch/flashinfer pins so they need isolation either way, containers
are cheaper on disk than two separate torch+CUDA installs, and it means the
committed ``docker/compose.full.yml`` is the configuration that was actually
measured rather than an untested file shipped for appearances.

Three things this module refuses to do implicitly, each because getting it wrong
silently corrupts results:

* **Run an unpinned image.** A ``:latest`` or ``nightly`` tag moves underneath
  you and makes every earlier result unreproducible. The resolved digest is
  required and is recorded into each run.
* **Trust the GPU index.** ``--gpus '"device=N"'`` selects the host GPU N but
  renumbers it to 0 inside the container. Identity is asserted by UUID.
* **Assume the kernel.** The startup log is parsed and the selected kernel
  checked against expectation before any measurement begins.
"""

from __future__ import annotations

import subprocess
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from llmbench.engines.kernels import assert_kernel

__all__ = ["EngineHandle", "EngineLaunchSpec", "EngineProcess", "EngineStartupError"]


class EngineStartupError(RuntimeError):
    """The engine failed to become ready, or came up misconfigured."""


@dataclass(frozen=True, slots=True)
class EngineLaunchSpec:
    """Everything needed to start one engine configuration."""

    config_id: str
    image: str
    tag: str
    #: Required. Resolved with `docker image inspect`; never a moving tag.
    image_digest: str
    model_hf_id: str
    model_revision: str
    gpu_index: int
    port: int
    max_model_len: int
    gpu_memory_utilization: float
    max_num_seqs: int
    #: Host HF cache, bind-mounted read-only so the already-downloaded weights
    #: are reused instead of re-pulled into each image.
    hf_cache_dir: Path
    expected_kernel: str | None
    kernel_log_patterns: Mapping[str, str]
    extra_args: Mapping[str, str] = field(default_factory=dict)
    startup_timeout_s: int = 900
    health_path: str = "/health"
    metrics_path: str = "/metrics"

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def container_name(self) -> str:
        return f"llmbench-{self.config_id}"


@dataclass(frozen=True, slots=True)
class EngineHandle:
    """A running, verified engine."""

    spec: EngineLaunchSpec
    container_id: str
    engine_version: str
    #: Kernel actually selected, parsed from the startup log.
    selected_kernel: str | None
    startup_log: str
    startup_duration_s: float

    @property
    def base_url(self) -> str:
        return self.spec.base_url


def _docker(args: Sequence[str], *, timeout: int = 120) -> str:
    """Run a docker command.

    Uses sudo because the invoking user's docker group membership only takes
    effect on a new login session; a benchmark should not require a re-login to
    be reproducible.
    """
    result = subprocess.run(
        ["sudo", "docker", *args],
        capture_output=True,
        text=True,
        check=True,
        timeout=timeout,
    )
    return result.stdout.strip()


def resolve_digest(image: str, tag: str) -> str:
    """Resolve a tag to its immutable ``sha256:`` digest.

    Raises:
        EngineStartupError: If the image is not present locally.
    """
    ref = f"{image}:{tag}"
    try:
        out = _docker(["image", "inspect", ref, "--format", "{{index .RepoDigests 0}}"])
    except subprocess.CalledProcessError as exc:
        msg = f"image {ref} not available locally: {exc.stderr.strip()}"
        raise EngineStartupError(msg) from exc

    if "@" not in out:
        msg = f"could not resolve a digest for {ref}; got {out!r}"
        raise EngineStartupError(msg)
    return out.split("@", 1)[1]


class EngineProcess(ABC):
    """Common container lifecycle; subclasses supply engine-specific argv."""

    name: str

    @abstractmethod
    def server_args(self, spec: EngineLaunchSpec) -> list[str]:
        """Engine-specific server arguments."""

    @abstractmethod
    def parse_version(self, startup_log: str) -> str:
        """Extract the engine version from its startup output."""

    def docker_args(self, spec: EngineLaunchSpec) -> list[str]:
        return [
            "run",
            "-d",
            "--rm",
            "--name",
            spec.container_name,
            # Selects host GPU N; inside the container it appears as index 0.
            "--gpus",
            f'"device={spec.gpu_index}"',
            "-p",
            f"{spec.port}:{spec.port}",
            "-v",
            f"{spec.hf_cache_dir}:/root/.cache/huggingface:ro",
            # vLLM and SGLang both need more than Docker's default 64 MB of
            # shared memory for their worker IPC.
            "--shm-size",
            "16g",
            "--ipc",
            "host",
            f"{spec.image}@{spec.image_digest}",
        ]

    def start(self, spec: EngineLaunchSpec) -> EngineHandle:
        """Launch, wait for health, then verify GPU identity and kernel.

        Raises:
            EngineStartupError: On timeout, wrong GPU, or kernel mismatch.
        """
        began = time.perf_counter()
        self.stop(spec)  # clear any stale container from an aborted run

        container_id = _docker([*self.docker_args(spec), *self.server_args(spec)])
        try:
            self._await_health(spec, container_id)
            log = self.logs(spec)
            self._assert_gpu_identity(spec)
            selected = assert_kernel(
                log,
                spec.expected_kernel,
                spec.kernel_log_patterns,
                config_id=spec.config_id,
            )
        except Exception:
            # Capture the log before teardown; a container that dies taking its
            # diagnostics with it costs a debugging cycle per failure.
            failure_log = self.logs(spec)
            saved = self._persist_failure_log(spec, failure_log)
            self.stop(spec)
            # The tail alone is often useless: a repeated warning can fill it
            # and push the real exception out of view. The full log is written
            # to disk and the path reported.
            raise EngineStartupError(
                f"[{spec.config_id}] engine failed to start cleanly. "
                f"Full container log: {saved}\n"
                f"Last 1500 chars:\n{failure_log[-1500:]}"
            ) from None

        return EngineHandle(
            spec=spec,
            container_id=container_id,
            engine_version=self.parse_version(log),
            selected_kernel=selected,
            startup_log=log,
            startup_duration_s=time.perf_counter() - began,
        )

    @staticmethod
    def _persist_failure_log(spec: EngineLaunchSpec, log: str) -> Path:
        """Write the full container log so a failure is diagnosable later.

        Containers are removed on failure, taking their logs with them, and a
        tail is often useless — a repeated warning can fill it and push the real
        exception out of view, which is exactly what happened with the
        read-only HF cache warnings.
        """
        path = Path("results/logs") / f"{spec.config_id}.failure.log"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(log)
        except OSError:
            return Path("<unwritable>")
        return path

    def _await_health(self, spec: EngineLaunchSpec, container_id: str) -> None:
        deadline = time.monotonic() + spec.startup_timeout_s
        url = f"{spec.base_url}{spec.health_path}"

        while time.monotonic() < deadline:
            if not self._is_running(spec):
                msg = f"container {container_id[:12]} exited during startup"
                raise EngineStartupError(msg)
            try:
                if httpx.get(url, timeout=5.0).status_code == httpx.codes.OK:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(3.0)

        msg = (
            f"[{spec.config_id}] engine not healthy after {spec.startup_timeout_s}s. "
            f"Weight loading plus CUDA graph capture can be slow on a cold cache."
        )
        raise EngineStartupError(msg)

    def _assert_gpu_identity(self, spec: EngineLaunchSpec) -> None:
        """Confirm the container really got the intended physical GPU.

        Compares UUIDs rather than indices: the container sees its GPU as index
        0 regardless of which host device it was given, so an index comparison
        would pass even when the wrong card was attached.
        """
        from llmbench.telemetry.gpu import resolve_uuid  # noqa: PLC0415

        expected = resolve_uuid(spec.gpu_index)
        actual = (
            _docker(
                [
                    "exec",
                    spec.container_name,
                    "nvidia-smi",
                    "--query-gpu=uuid",
                    "--format=csv,noheader",
                ]
            )
            .splitlines()[0]
            .strip()
        )

        if actual != expected:
            msg = (
                f"[{spec.config_id}] container was given GPU {actual}, expected "
                f"{expected} (host index {spec.gpu_index}). Telemetry and workload would "
                f"be describing different devices."
            )
            raise EngineStartupError(msg)

    def _is_running(self, spec: EngineLaunchSpec) -> bool:
        try:
            state = _docker(
                ["inspect", "-f", "{{.State.Running}}", spec.container_name], timeout=30
            )
        except subprocess.CalledProcessError:
            return False
        return state == "true"

    def logs(self, spec: EngineLaunchSpec) -> str:
        try:
            result = subprocess.run(
                ["sudo", "docker", "logs", spec.container_name],
                capture_output=True,
                text=True,
                check=False,
                timeout=60,
            )
        except subprocess.SubprocessError:
            return ""
        return result.stdout + result.stderr

    def stop(self, spec: EngineLaunchSpec) -> None:
        """Tear down; safe to call when nothing is running."""
        subprocess.run(
            ["sudo", "docker", "rm", "-f", spec.container_name],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
