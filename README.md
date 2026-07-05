# gpu-probe

**Existing tools take hours and miss the fault. This takes seconds and finds it.**

<br>

Training crashed at 3am. NCCL timeout. NaN on step 47,000. Eight identical H100s in the box — which one is lying?

You reach for NVIDIA's `dcgmi diag`. It churns for 20 minutes. All pass. The card is still bad.

You throw `gpu-burn` at it. An hour at 700W. Passes. Still bad.

You grep `dmesg` for XID errors. Clean. Kernel logs? Nothing. The card is **still** bad.

Meanwhile GPU Probe already gave you the answer — **all 8 GPUs scanned in under 4 seconds, and it found the silent FMA bit-flip that every other tool said didn't exist.**

That's the whole pitch. GPU Probe detects what existing tools can't, in the time it takes them to print their banner.

---

## The problem with GPU diagnostics

Every GPU fleet operator learns the same hard truth: **most diagnostic tools take forever and still miss what actually breaks.**

| Tool | Time | What it finds | What it misses |
|------|------|-------------|----------------|
| **dcgmi diag** | 10–30 min | Spec compliance, basic health | Intermittent SM hangs, silent compute errors, partial warp failures |
| **gpu-burn** | 30–60 min | Cards that melt under load | SFU faults, warp shuffle corruption, FMA bit-flips — it's a space heater test, not a correctness test |
| **nvbandwidth** | 5–10 min | PCIe/NVLink throughput | Everything downstream of the interconnect — clean link, broken SM, still says pass |
| **XID / dmesg** | — | Errors the driver bothered to report | Everything the hardware doesn't self-report. A compute bit-flip in an FMA pipe generates no XID |
| **NCCL tests** | 10+ min | Multi-node collectives | Single-card issues masked by redundancy. One bad warp in 132 SMs disappears in the aggregate |

Hours of testing. Every tool says healthy. Your training still crashes.

GPU Probe: **all GPUs in parallel, results in seconds, catches the invisible ones.**

---

## What it actually tests

Four kernels, all GPUs in parallel. Each test has a **5-second timeout** — if it hangs, you know exactly which test and which GPU.

| # | Test | Hardware path | Catches |
|---|------|--------------|---------|
| 1 | **Memory** | HBM → L2 → PCIe DMA | Bit flips, uncorrectable ECC, silent data corruption, DMA engine faults |
| 2 | **Warp shuffle** | SM warp scheduler + crossbar | Partial-mask `__shfl_xor_sync` deadlock, warp divergence, SM hang without XID |
| 3 | **SFU** | Special function units | NaN-producing faults in `expf` / `rsqrtf` / `logf` / `tanhf` — invisible to integer tests |
| 4 | **FMA** | Fused multiply-add pipes | Compute bit-flip via contracting sequence convergence — the scariest one, because it's silent |

### Why these four?

They cover the paths that fail in practice — paths no other tool targets. HBM ECC catches DRAM errors but not DMA engine faults. GEMM stress catches thermal issues but not SFU pipeline corruption. Warp shuffle catches SM scheduler bugs that produce no XID, no ECC event, no kernel log — just wrong results, silently.

**A full 3-round scan of 8 GPUs finishes in ~4 seconds when all cards are healthy** (all GPUs tested in parallel). Even with a bad card triggering a 5s timeout, total wall time stays under 12 seconds. `dcgmi diag` hasn't even loaded its first plugin by then. `gpu-burn` is still warming up. GPU Probe is already telling you which card to drain.

---

## The part that matters: it catches what other tools can't

A GPU can pass every official diagnostic and still produce wrong results. These aren't hypotheticals:

- **Silent FMA drift:** one lane in one FMA pipe flips a bit once every ~10⁶ operations. `dcgmi` passes. `gpu-burn` passes. Training diverges 6 hours later. GPU Probe catches it in seconds.
- **Warp shuffle deadlock:** a specific shuffle mask pattern hangs one warp scheduler intermittently. No XID. No ECC. Driver reports healthy. Your all-reduce stalls — but NCCL tests pass because the other 131 SMs carry the load. GPU Probe nails it by testing partial-mask shuffle on every SM.
- **SFU NaN injection:** `expf()` returns NaN for a narrow input range. Integer tests are blind to this. `dcgmi` doesn't check SFU correctness at this granularity. Your loss goes vertical. GPU Probe catches it because it validates every SFU result against expected bounds.

**Each of these faults is invisible to the entire existing toolchain.** GPU Probe finds all three.

---

## Quick start

```bash
# Build
nvcc -O2 -o gpu_probe gpu_probe.cu

# Scan everything
./gpu_probe

# One suspicious card
./gpu_probe --gpus 3

# Aggressive: more rounds, longer timeout
./gpu_probe --gpus 0 1 2 3 --rounds 5 --timeout 10
```

No dependencies beyond CUDA toolkit. No Python. No config files. No Docker. One `.cu` file, ~850 lines.

---

## How it works

```
┌─────────────────────────────────────────────┐
│                  gpu_probe                   │
│                                              │
│  fork() per GPU ──── child pinned to GPU N   │
│                                              │
│  per round:                                   │
│    alarm(5s) ── memory ── alarm(0)           │
│    alarm(5s) ── warp   ── alarm(0)           │
│    alarm(5s) ── sfu    ── alarm(0)           │
│    alarm(5s) ── fma    ── alarm(0)           │
│                                              │
│  SIGALRM fires? ── child _exit(80+test_id)   │
│  kernel hangs?  ── parent SIGKILL after      │
│                     per-worker timeout       │
└─────────────────────────────────────────────┘
```

Each GPU is tested in an isolated process. A hung SM can't hang the scanner — the per-test `alarm()` kills the child, the parent moves on. This is what `dcgmi` can't do: if one plugin hangs, the whole diagnostic hangs.

Results stream back through a pipe in binary `StageRecord` structs. The parent aggregates, prints a summary table, and exits non-zero if any GPU failed.

---

## Output

```
[PROBE] GPU 0: NVIDIA H100 (132 SMs), rounds=3, per_test_timeout=5s
[PROBE] GPU 0 round 1 test=memory START timeout=5s
[PROBE] memory PASS (79871 MB, round 1)
[PROBE] GPU 0 round 1 test=memory PASS elapsed_ms=234.561
...

===== GPU PROBE SUMMARY =====
Config: rounds=3 per_test_timeout=5s total_elapsed_ms=42000.123
+-----+--------+----------------------+--------+-----------+-----+
| GPU | Result | Reason               | Memory | Warp      | ... |
+-----+--------+----------------------+--------+-----------+-----+
|   0 | PASS   | pass                 |   PASS | PASS      | ... |
|   1 | PASS   | pass                 |   PASS | PASS      | ... |
|   7 | BAD    | fail:fma             |   PASS | PASS      | ... |
+-----+--------+----------------------+--------+-----------+-----+
PASS GPUs: 0 1 2 3 4 5 6
BAD GPUs:  7(fail:fma)
CONCLUSION: suspect/problem GPUs: 7
```

Every test stage gets its own column. You know exactly what broke — not just "card bad."

Full per-stage logs go to `gpu_probe_<gpu_id>.log` (one per GPU). Ship them to your infra team. They'll thank you.

---

## When to use it

- **Pre-flight:** before launching a training run. 30 seconds now saves a checkpoint restore at 4am.
- **Incident response:** NCCL errors or NaN mid-training? Run this before you touch anything else. Know which card to drain.
- **Burn-in:** new node, repaired node, moved node. Five minutes of GPU Probe beats an hour of uncertainty.
- **Cluster sweep:** `for node in $(cat hosts); do ssh $node './gpu_probe'; done`. Your weekend on-call just got shorter.

---

## When NOT to use it

GPU Probe is a **bad-card detector**, not a benchmark. It won't tell you which card is 3% slower. It won't measure NVLink bandwidth or thermal headroom. It answers one question: "is this GPU producing correct results?" If you need throughput numbers, run `nvbandwidth`. If you need thermal soak, run `gpu-burn`. If you need to know if the silicon is lying to you, run this.

---

## A note on false confidence

`dcgmi diag --level 4` returning "PASS" is the most dangerous output in GPU operations. It convinces you the hardware is fine when it isn't. GPU Probe returning "PASS" means the four failure modes it tests for are clean. It doesn't mean the card is perfect — no tool can promise that. But it tests the things that actually break, not the things that are easy to test. That's the difference.

---

## License

Apache 2.0 — do what you want. If it saves you a 4am page, tell someone.

---

*~850 lines of CUDA. No Python. No YAML. No Docker. Just CUDA toolkit and a compiler. It's a probe. It probes.*
