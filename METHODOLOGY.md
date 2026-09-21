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

**Why not closed-loop — and why the usual reason turned out to be wrong here.** A
fixed-concurrency generator only issues a new request when a prior one completes. The textbook
objection is **coordinated omission**: when the server stalls, the generator stalls with it, the
slow period is under-sampled, and the reported tail is optimistic. That was this document's
claim, and the closed-loop exhibit (`vllm-bf16-closed-loop`) was built to measure the size of
the error.

It measured the opposite sign. At throughput-matched points the closed loop never understates
the open-loop p99: below the knee the two are indistinguishable, and at 5–7 rps the closed loop
reports a tail **3.3–3.7× worse** (REPORT.md §6). A continuous-batching engine below saturation
has no stalls for coordinated omission to hide. What a closed loop does instead is hold
occupancy at a constant maximum, so every new request's prefill competes with a full batch of
decodes; Poisson arrivals at the same mean let occupancy fluctuate, and requests that land in
a lull get fast prefill.

The conclusion survives; the reason had to be replaced. Open-loop is correct not because a
closed loop flatters the server but because a closed loop's offered load is a *consequence of
the server's speed* — the generator and the thing under test are coupled, and a coupled
generator cannot measure the server against any load a deployment would actually receive. On
this stack that coupling produced a pessimistic tail rather than an optimistic one. Either way
it is not the tail a real arrival process would see.

The exhibit reuses `fire_one`, the open-loop generator's own request function, rather than
reimplementing it. Two generators with two SSE parsers and two TTFT definitions would differ in
more than their arrival process, and the exhibit would stop isolating the one variable it
exists to isolate. Its concurrency is chosen so achieved throughput matches the open-loop run
it is compared against, because a tail comparison across different offered loads measures the
load rather than the generator.

**Schedule determinism.** The full arrival schedule and prompt list are generated from a fixed
seed *before* the run begins. Every configuration therefore faces a byte-identical offered load,
which removes generator-side variance from all cross-configuration comparisons.

**Client saturation is detected, not assumed away.** At high λ the *client* can become the
bottleneck, at which point the run silently measures the load generator instead of the server.
Every request records `dispatch_lag` = actual − scheduled dispatch time. If p99 dispatch lag
exceeds threshold the run is marked `CLIENT_SATURATED` and excluded from headline results. The
record is kept, because the rate at which a harness runs out of headroom is itself a finding.

Measured dispatch lag, BF16, p99 across three repeats:

| Offered | 1 rps | 2 rps | 4 rps | 8 rps | 12 rps | 24 rps |
|---|---|---|---|---|---|---|
| p99 lag | 5 ms | 7 ms | 14 ms | 49 ms | 55 ms | 185 ms |
| as % of TTFT p95 | 3.3 % | 5.1 % | 5.6 % | 0.27 % | 0.04 % | 0.04 % |

The client stayed well inside its budget throughout. Note that lag matters
*most* at low rates, where it is a larger share of a small TTFT — the opposite
of the intuition that high load is where the harness breaks. See
[Saturation is not client failure](#saturation-is-not-client-failure).

---

## 3. Workload realism

Fixed 128-in/128-out is the classic tell of an unserious benchmark: real traffic has a long tail
and length distribution changes batching behaviour completely.

- **Primary:** input/output lengths sampled from **ShareGPT** conversations — real
  human/assistant turns, and the same source vLLM's own `benchmark_serving.py` uses, so numbers
  remain comparable to published work.
- **Secondary:** an **Azure LLM Inference Trace** replay, supplying real production *arrival
  timestamps* rather than assumed Poisson. Measured as five independent non-overlapping
  windows against five independent Poisson seeds, interleaved: p99 TTFT **+4 ms against a
  ±21 ms interval** — indistinguishable. The trace was characterised first, and the replay
  agreed with the characterisation.

  **The published Azure conversation trace is very close to Poisson at this timescale.** In a
  180 s window matched to 4 req/s, its inter-arrival times have a squared coefficient of
  variation of **1.02** against a Poisson process's 1.00. The expectation going in was that
  real traffic would be markedly burstier; measured, at the rate and window this benchmark
  operates on, it is not.

  That measurement is what the metric is for, and getting it required fixing the metric first.
  An earlier version divided inter-arrival variance by the mean, which is not dimensionless —
  for an Exponential distribution it *equals* the mean, so it reads 1.0 only at exactly 1 req/s
  and 0.25 at 4 req/s, and a trace would appear to become less bursty purely by arriving
  faster. It is now variance over mean squared, whose Poisson reference is 1.0 at any rate.

  Window selection matters as much as the metric. Windows are contiguous, because sampling
  requests from across the trace would destroy the temporal correlation that makes it bursty
  and leave a reordered Poisson-ish process wearing a trace's name; and they must fit entirely
  inside the trace, because a truncated window near the end holds few requests, reports a rate
  far below the trace's real one, and would be selected as the closest match to any low target
  while offering a fraction of the intended load.

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

**Chassis thermal coupling — measured, and smaller than feared.** The measurement GPU shares a
passively-cooled chassis with a second A40. When the neighbouring card is under load it raises
inlet air temperature on the measurement device. Mitigations: measurement is pinned to a single
GPU; clocks are locked; throttle telemetry is captured per run; and every record carries a
`neighbor_gpu_busy` flag so no result can quietly lose its asterisk.

66 of the 168 runs were measured beside a busy neighbour, including **both SGLang
configurations in their entirety** — which puts the confound directly on the engine axis, where
it matters most. The stamp made it possible to check rather than argue about:

| | runs | temp mean | temp max | SM clock | power | throttled samples |
|---|---|---|---|---|---|---|
| busy neighbour | 60 | 68.8 °C | 72 °C | 1669 MHz | 287.9 W | 0 |
| quiet chassis | 63 | 69.9 °C | 72 °C | 1662 MHz | 289.7 W | 0 |

Runs beside a busy neighbour were **1.1 °C cooler at 7 MHz higher clocks** — the opposite
direction to the feared effect, and small. No record throttled; the only reason ever observed
anywhere in the sweep is `sw_power_cap`, which is the card's own 300 W budget rather than
thermal coupling. `vllm-bf16` crosses the quiet-to-busy boundary between 4 and 5 rps with no
discontinuity in TTFT, TPOT or throughput.

This does not prove the coupling is always negligible; it bounds it for these runs, on this
chassis, at this ambient. The stamp stays on the records so a reader can check the claim
instead of taking the paragraph on trust.

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
| Shared HF cache | Three quantized checkpoints **deleted mid-sweep** by another user reclaiming disk | The cache is bind-mounted read-only, by design, so a missing checkpoint cannot self-heal by downloading. Preflight now asserts the pinned repository *and* revision are present before any engine launches. |

### A shared machine will delete your inputs

Between two passes of the sweep, the three quantized checkpoints vanished from the host
Hugging Face cache — another user on the box reclaiming disk. Two configurations then failed at
container start, and the failure said only `No such container`, because the engines run with
`docker run --rm` and Docker reaps a container the instant it dies. The flag intended to tidy
up had destroyed the only evidence of what went wrong. Re-running the identical command without
it produced the real error immediately:

```
OSError: [Errno 30] Read-only file system:
  '/root/.cache/huggingface/hub/models--RedHatAI--...w8a8'
```

Three things came out of this, all of them cheap and none of them obvious beforehand:

* **`--rm` is incompatible with diagnosability.** Teardown is explicit in `stop()` anyway, so
  the flag bought nothing and cost a debugging cycle.
* **Preflight checks the revision, not just the repository.** A cached repo at the wrong commit
  is not the pinned checkpoint, and accepting it would quietly measure different weights.
* **The check runs over the whole matrix before the first engine starts.** A missing checkpoint
  is fatal either way; discovering it at minute zero costs a second, and discovering it after
  the first configuration's pass costs an hour.

Nothing measured was lost — the affected records predate the deletion. Because model revisions
are pinned to commit SHAs rather than branch names, the re-downloaded weights are provably the
same ones, which is the reproducibility argument this repository makes, tested against an
accident rather than asserted.

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

### Saturation is not client failure

Beyond capacity, the two look alike and are not the same thing. Measured on
this hardware, achieved request rate pins at **~6.15 req/s** no matter what is
offered — 8, 12, 16 and 24 rps all achieve it. Past that point the queue grows
without bound and TTFT becomes a function of how long the run lasted (129 s at
12 rps, 456 s at 24 rps) rather than of the offered load.

The first version of the validity guard labelled those runs `CLIENT_SATURATED`,
because dispatch lag crossed its absolute and inter-arrival thresholds. But lag
was **0.04 % of the measured TTFT**. Blaming the harness for the server's
capacity limit would have been exactly the misattribution this document
criticises elsewhere.

Two consequences, both now enforced:

* **Client saturation is judged relative to what is being measured.** A 55 ms
  dispatch lag contaminates a 140 ms TTFT and is irrelevant against 129 s.
* **`OVERSUBSCRIBED` outranks `CLIENT_SATURATED`.** Past capacity the two are
  not independent: a server that cannot keep up leaves thousands of requests in
  flight, and that alone induces lag in any client. The lag is a symptom.

Oversubscribed runs are excluded from the frontier but kept and reported. They
are how the saturation point is located.

### What the smoke gate cannot catch

`make bench-smoke` runs the pipeline end to end in ~6 minutes and is the gate
before the full matrix. It cannot, in principle, cover everything the sweep hits.

The smoke config runs 30 s at rates 1 and 4, drawing ~150 requests. The full
sweep draws 4370 at 24 rps. When the sweep first ran, it died at its seventh
measurement on a ShareGPT conversation with a 4634-token prompt against
`max_model_len=4096` — a sample from the tail of a heavy-tailed distribution,
reachable only at the sample sizes the real run uses.

A gate that runs a *smaller version* of the workload cannot surface failures
that live in the tail. That is a limitation of the gate, not a bug in it, and
it is the reason the sweep writes one validated JSON per measurement: when the
tail does bite, it costs one measurement rather than the run.

### A worked example of this repository's own thesis

The first complete quality result parsed cleanly and looked plausible. Checked
against published numbers rather than accepted, it was wrong by 32 points:

| Metric | Measured | Published | |
|---|---|---|---|
| WikiText-2 PPL | 6.3546 | ~6.2–6.5 | ok |
| GSM8K strict | 0.7286 | ~0.76–0.84 | ok |
| IFEval prompt-strict | **0.4603** | ~0.78–0.80 | **32 points low** |

IFEval is a 0-shot *instruction-following* benchmark. Served through raw
`/v1/completions` with no chat template, an instruct model continues text rather
than acting as an assistant. Routing it through `/v1/chat/completions` with the
template moved prompt-level strict accuracy from **0.4603 to 0.7000** and
instruction-level from 0.6019 to 0.7937.

The number was not noise and not a bug in the sense a test would catch — it
passed type checking, linting and 359 tests. It was a *plausible* number
produced by the wrong experiment, and only comparison against an external
reference exposed it. That is the failure mode this repository is about, and it
happened here.


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

One configuration is additionally measured with vLLM's own harness, `vllm bench serve`, which
ships inside the pinned engine image. Agreement between an independent implementation and this
one is stronger evidence of correctness than any amount of self-written unit testing: every
other check here is self-referential — the percentile function against numpy, the arrival
process against a KS test, TTFT against internal consistency — and all of it can be true while
the harness measures the wrong thing consistently.

**Run inside the engine's own container**, so there is no second Python environment to install,
drift, or explain. The comparison is then between two harnesses rather than two environments.

**Matched deliberately:** same live server, model, seed, request rate, ShareGPT corpus,
`/v1/completions` endpoint, `ignore_eos`, and `--burstiness 1.0` stated explicitly rather than
left to a default — leaving the arrival process implicit is the easiest way to compare two
different workloads and then call the disagreement a bug.

**Deliberately not matched:** the dispatch loop, the SSE parsing, the TTFT definition and the
percentile implementation. Those are the subject of the comparison; making them match would
defeat it.

Agreement is judged at ±5 %. That is loose enough to absorb the difference between two separate
measurements against a live server — they are not replays of one run — and tight enough that a
real defect in either dispatch loop or percentile implementation would show. A larger
disagreement is a finding, not noise.

**Result.** With both harnesses on constant 256/256-token requests against the same live
server, every latency metric agrees within 5 %: TTFT mean +3.5 %, TTFT p99 +4.9 %, TPOT mean
−2.6 %, TPOT p99 −1.6 %, E2E −2.5 %. Output throughput sits at −5.1 %, a hair outside.

The first attempt, with each harness sampling ShareGPT its own way, disagreed on E2E by +63 %
and throughput by +50 % while agreeing on TTFT mean to 0.3 %. That shape was the diagnosis:
ours drew 303 output tokens per request and upstream 192 from the same corpus, and the
throughput ratio predicted from the lengths alone (1.494) matched the measured one (1.500).
The harnesses agreed on everything they measured the same way and differed by exactly the
workload. Both tables are in REPORT.md §6, because a reader shown only the clean result would
not know the unclean one existed.

---

## 10. What more hardware would buy

Now answerable against measured results rather than guessed at, because the sweep found a
specific mechanism worth testing elsewhere.

**An Ada or Hopper card, to make the FP8 axis real.** This is the strongest candidate, and the
results say why. The headline finding is that the winning format changes with load: W4A16 buys
bandwidth only, so it wins bandwidth-bound decode and collapses on compute-bound prefill, while
W8A8 accelerates prefill arithmetic on real INT8 tensor cores and therefore sustains the highest
rate. On sm_89/sm_90 the W8A8 role would be filled by FP8 — same structural argument, different
numeric format, and quality damage that is typically smaller than INT8's. Whether the crossover
sits at the same place, or moves, is the obvious next experiment. It cannot be run here: the
A40 is sm_86, and vLLM will silently dequantize an FP8 checkpoint to FP16 rather than refuse,
which is why FP8 is excluded here rather than measured badly.

**A higher-bandwidth part of the same generation, to test the mechanism directly.** The bytes-read
model predicted INT8 decode within 1.6 % on this card. If the model is right rather than merely
fitted, the *same* prediction should hold on an A100 or H100 while the absolute advantage of
weight-only quantization narrows, because those parts are less bandwidth-starved relative to
their compute. That is a falsifiable prediction this repository cannot test with one GPU model,
and it is the single measurement that would most strengthen or break the central claim.

**NVLink, to make tensor parallelism measurable.** The two A40s here are joined by a PCIe host
bridge with no NVLink, so a TP=2 run would measure the interconnect rather than the model. On an
8B model that is expected to be a net loss and was cut for exactly that reason; on a 70B, where
TP is not optional, the interconnect becomes the thing worth measuring rather than the thing
contaminating the measurement.

**A machine that is not shared.** Two of the incidents recorded in §5a — the deleted checkpoints
and the busy-neighbour stamp on 66 of the first 168 runs — are artifacts of a multi-tenant box rather than
of the method. Neither invalidated a result, because both were detected and bounded, but both
cost time that dedicated hardware would not have.
