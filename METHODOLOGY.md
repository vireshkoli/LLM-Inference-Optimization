# Methodology

This document exists to let a reader decide whether to believe the numbers. It states what was
measured, how load was generated, what was held constant, what was discarded, and — most
importantly — what is wrong with the setup anyway.

> **Status:** methodology is fixed as of Phase 1 and is stated here *before* results exist, so
> it cannot be retrofitted to flatter them. Sections marked _(pending)_ are filled from measured
> data in Phase 8.

---

## 1. What is measured

Throughput is not a number, it is a curve. A result of the form "N tokens/sec" without a stated
load level and latency distribution is unfalsifiable. The core artifact is therefore a
**latency-vs-throughput curve per configuration**, swept across offered request rates until
saturation, with the knee identified from the data.

Metrics are kept separate, because reporting only end-to-end latency hides whether a
configuration is prefill- or decode-bound:

| Metric | Definition |
|---|---|
| **TTFT** | Arrival of the first streamed chunk **containing an actual token**, minus dispatch time |
| **TPOT / ITL** | Per-token inter-arrival time across the decode phase |
| **End-to-end latency** | Dispatch to final chunk |
| **Output token throughput** | Generated tokens ÷ measurement window |
| **Request throughput** | Completed requests ÷ measurement window |

Each is reported as mean, std, p50, p90, p95 and p99.

> **A trap worth naming.** The first SSE chunk from an OpenAI-compatible endpoint is frequently
> a role-only delta carrying no token. Timing to *that* chunk understates TTFT. TTFT here is
> time to first chunk with non-empty content, and `tests/test_stream.py` pins that behaviour.

---

## 2. Load generation — open-loop, and why

Load is generated **open-loop with Poisson arrivals**. Inter-arrival times are drawn from
`Exponential(1/λ)` and dispatched at their scheduled wall-clock instant *regardless of whether
earlier requests have returned*.

**Why not closed-loop.** A fixed-concurrency generator only issues a new request when a prior
one completes. When the server slows, the generator slows with it, so the slow period is
under-sampled — **coordinated omission**. The measured tail is then systematically optimistic,
and the tail is precisely the number anyone cares about. A closed-loop run is included in the
sweep (`vllm-bf16-closed-loop`) purely as an exhibit demonstrating the size of this error on
this hardware; it is never reported as a headline result.

**Schedule determinism.** The full arrival schedule and prompt list are generated from a fixed
seed *before* the run begins. Every configuration therefore faces a byte-identical offered load,
which removes generator-side variance from all cross-configuration comparisons.

**Client saturation is detected, not assumed away.** At high λ the *client* can become the
bottleneck, at which point the run silently measures the load generator instead of the server.
Every request records `dispatch_lag` = actual − scheduled dispatch time. If p99 dispatch lag
exceeds threshold the run is marked `CLIENT_SATURATED` and excluded from headline results. The
record is kept, because the rate at which a harness runs out of headroom is itself a finding.

_(pending: measured dispatch-lag distributions per rate)_

---

## 3. Workload realism

Fixed 128-in/128-out is the classic tell of an unserious benchmark: real traffic has a long tail
and length distribution changes batching behaviour completely.

- **Primary:** input/output lengths sampled from **ShareGPT** conversations — real
  human/assistant turns, and the same source vLLM's own `benchmark_serving.py` uses, so numbers
  remain comparable to published work.
- **Secondary:** an **Azure LLM Inference Trace** replay, which supplies real production
  *arrival timestamps* rather than assumed Poisson. Poisson is itself a modelling assumption;
  this run quantifies what that assumption costs at the tail.

**Output length is enforced**, via `max_tokens` set to the sampled length together with
`ignore_eos=True`. Without this, different quantization levels stop at different points and the
comparison silently spans different workloads. The cost of this choice is realism — real traffic
does stop at EOS — and it is a deliberate trade of realism for control.

---

## 4. Steady state

- Warmup requests are issued and **discarded** before every measurement window.
- CUDA graph capture and any `torch.compile` work happen during engine startup, **outside** the
  measurement window.
- A settle/drain period separates consecutive rate points.
- GPU clocks are pinned with `nvidia-smi -lgc`; the policy is recorded in every result record,
  including when locking was *not* applied.
- Temperature, SM clock, power draw and `clocks_event_reasons.active` are sampled throughout
  each run. `throttled_fraction` travels with the result.

### Not every active event reason is a throttle

Measured on this hardware: an A40 at 96 % utilisation reports `sw_power_cap` **continuously**.
That is the card holding its 300 W budget — normal operation, not a fault. Treating any active
event reason as throttling would mark every loaded run `THERMAL_THROTTLED` and leave the entire
sweep unreportable.

Only `hw_slowdown`, `sw_thermal_slowdown`, `hw_thermal_slowdown` and `hw_power_brake_slowdown`
invalidate a measurement. Every other reason is still recorded and reported, so a reader can
see that `sw_power_cap` was continuously active without it condemning the run. Hiding it would
be as dishonest as counting it.

### A clock lock is not absolute on a power-capped card

Measured during a saturating run with a 1740 MHz lock applied:

| | |
|---|---|
| SM clock | median **1740 MHz**, minimum **1515 MHz** |
| Samples >30 MHz below the lock | **21 %** (18 of 85) |
| Power draw | median 115 W, max **302 W** against a 300 W cap |
| Samples at/above 290 W | **18 of 85** |

The correspondence is exact: every clock dip coincides with the power cap
binding. `nvidia-smi -lgc` removes *boost* variability but cannot defeat the
power budget — at peak load the card trades clock for watts.

This is reported rather than hidden, and it sets the verification threshold
honestly: demanding ~100 % clock adherence would fail every heavily-loaded run
for a reason that is physics, not a fault. The full clock distribution is stored
in each result so a reader can judge it directly instead of trusting a boolean.

### Verifying the clock lock

`nvidia-smi -lgc` sets a locked clock range that this driver exposes **no query field for**.
The obvious-looking `clocks.applications.graphics` reports the *default* applications clock,
which on an A40 equals the max (1740 MHz) whether or not a lock is active — so reading it
reports "locked" for an unlocked card.

The lock is therefore recorded by the script that applies it and cross-checked behaviourally.
That check is only meaningful under load: measured directly, an idle A40 with a 1740 MHz lock
applied still drops to 210 MHz, and only holds 1740 MHz once a CUDA context exists. A preflight
probe on an idle card therefore returns "unknown", never "failed"; the real verification runs
against clocks sampled during the measurement window.

---

## 5. Known confounds

**Chassis thermal coupling.** The measurement GPU shares a passively-cooled chassis with a
second A40. When the neighbouring card is under load it raises inlet air temperature on the
measurement device. Mitigations: measurement is pinned to a single GPU; clocks are locked;
throttle telemetry is captured per run; every record carries a `neighbor_gpu_busy` flag; and a
**drift canary** re-runs the first configuration at the end of the sweep. If canary and original
agree within noise, environmental drift across the sweep is bounded.

**The sweep is never parallelised across both GPUs.** Doing so would halve wall-clock and
reintroduce exactly the shared-airflow, shared-PCIe, shared-vCPU confound the rest of this
section works to control. This is a deliberate choice to spend time rather than validity.

**Engine configuration parity is imperfect.** vLLM's `--gpu-memory-utilization` and SGLang's
`--mem-fraction-static` are not semantically identical, and `max_num_seqs` maps to
`--max-running-requests`. These are documented rather than smoothed over; see
`configs/engines/sglang.yaml`.

**Quantized checkpoints come from different publishers.** The BF16, INT8 W8A8 and GPTQ INT4
weights share a calibration lineage; the AWQ INT4 checkpoint does not. Some of any observed
AWQ-vs-GPTQ quality difference is attributable to calibration rather than to the quantization
scheme.

**Greedy decoding is not bitwise-deterministic across batch sizes.** Reduction order in fused
kernels varies with batch shape, so identical prompts can produce different tokens at different
concurrency. Quality evaluations therefore run at a fixed low concurrency, recorded in each
`QualityResult`.

---

## 5a. Measured environment facts

Established by running the stack, not assumed at planning time. Several contradict what the
documentation or prior experience would suggest.

| Fact | Value | Why it matters |
|---|---|---|
| Container GPU renumbering | `--gpus '"device=1"'` → index **0** inside the container | Telemetry keyed on index would record the neighbouring GPU. Identity is asserted by **UUID**. |
| KV cache, BF16, `max_model_len=4096` | **195,760 tokens / 23.9 GiB** | ~20 % below the 245k estimated from a naive 128 KiB/token calculation. KV is still not the bottleneck. |
| Engine startup | ~220 s to healthy; 54.2 s engine init of which **29.6 s is compilation**, 6 s CUDA graph capture | All of it precedes the measurement window. Amortised by starting the engine once per configuration. |
| vLLM image on disk | **18.9 GB** (10.3 GB compressed) | Roughly double the planning estimate; the disk budget was revised accordingly. |
| Driver ceiling | 570.133.07 → CUDA 12.x only | SGLang **must** be a `-cu129` build. A `-cu130` image needs driver ≥ 580 and fails at container start with a CUDA init error that never mentions the driver. |

### Prometheus metric names were verified, not assumed

vLLM's V1 engine renamed several metrics. Panels built on the older names render empty, which
looks indistinguishable from an idle server:

| Commonly documented | Actual in v0.26.0 |
|---|---|
| `vllm:gpu_cache_usage_perc` | `vllm:kv_cache_usage_perc` |
| `vllm:time_per_output_token_seconds` | `vllm:request_time_per_output_token_seconds` |

The verified names are recorded in `configs/engines/vllm.yaml` and are what the committed
Grafana dashboards query.

### The bandwidth-bound premise, measured rather than inferred

This benchmark's central claim is that the A40 is memory-bandwidth-bound during
decode (~696 GB/s against ~150 TFLOPS BF16), and that weight-only quantization
should therefore pay off more here than on the higher-bandwidth GPUs most public
benchmarks use. That is a spec-sheet inference until something measures it.

DCGM profiling counters, sampled during a live run at moderate load:

| Counter | Value |
|---|---|
| `DCGM_FI_PROF_DRAM_ACTIVE` | **0.89** — memory bandwidth 89 % saturated |
| `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE` | **0.21** — tensor cores 21 % busy |

Decode on this card is unambiguously bandwidth-bound, not compute-bound. As
offered load rises and batches grow, the ratio shifts as theory predicts —
at saturation DRAM activity fell to 0.69 while tensor activity rose to 0.58,
because a larger batch amortises each weight read across more tokens.

The premise behind the whole quantization axis is therefore established by
measurement before any quantized configuration is benchmarked.

### Harness cross-check against a live engine

The measured metrics are internally consistent, which is the strongest available evidence that
the timing code is correct. From a 4 rps run against vLLM serving Llama-3.1-8B BF16:

```
TPOT p50 × 63 tokens + TTFT p50  =  30.18 ms × 63 + 93.67 ms  =  1995 ms
E2E p50 (independently measured) =                               1997 ms
```

Output token count was exactly 64.0 on every request with `finish_reason=length`, confirming
`ignore_eos` makes every configuration perform identical work.

---

## 6. What is excluded, and why

**FP8 is not benchmarked.** Native FP8 arithmetic requires compute capability 8.9 (Ada) or 9.0
(Hopper). The A40 is 8.6. vLLM will nonetheless *load* an FP8 checkpoint on Ampere by
dequantizing to FP16 — the server starts, requests succeed, and the resulting number measures
weight-only compression with no compute speedup. Publishing that as "FP8 on A40" would be
precisely the class of error this repository exists to demonstrate against.

Also excluded: Machete kernels (sm_90 only), TensorRT-LLM (per-configuration engine compilation
would exceed the entire measurement budget), and llama.cpp (a different deployment class;
comparing it to vLLM under concurrent datacenter load would be a category error).

---

## 7. Kernel selection is asserted, not assumed

Both INT4 checkpoints available for Llama-3.1-8B carry `desc_act=true`. vLLM's `gptq_marlin`
supports act-order via a load-time permutation, at some runtime cost; AWQ-GEMM has no act-order
concept. A checkpoint that silently falls off the fast kernel path would quietly halve
throughput and corrupt the entire comparison.

The harness therefore parses engine startup logs, asserts the selected kernel against the
expectation declared in `configs/sweep.yaml`, and records the observed kernel into every result.
A mismatch fails the run at launch, not at analysis time.

---

## 8. Variance

Every (configuration, request-rate) point is run **≥3 times**. Mean and standard deviation are
reported, and every chart carries error bars. A benchmark without error bars invites the reader
to assume the author got lucky once.

---

## 9. Cross-validation

One configuration is additionally measured with vLLM's upstream `benchmark_serving.py`.
Agreement between an independent implementation and this harness is stronger evidence of
correctness than any amount of self-written unit testing.

_(pending: agreement figures)_

---

## 10. What more hardware would buy

_(pending — written against measured results in Phase 8)_

Candidates: an Ada or Hopper card to make the FP8 axis real rather than excluded; NVLink to make
tensor-parallel scaling measurable without the interconnect dominating; a second GPU generation
to test whether the bandwidth-bound quantization advantage predicted for the A40 actually
narrows on higher-bandwidth parts.
