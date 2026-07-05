/**
 * gpu_probe.cu — GPU bad-card detector
 *
 * 4 core tests covering the essential hardware paths:
 *   1. Memory: write/verify patterns (covers HBM + L2 + DMA path)
 *   2. Warp shuffle: partial-mask __shfl_xor_sync (detects SM hang)
 *   3. SFU: exp/rsqrt/log/tanh correctness (detects NaN-producing faults)
 *   4. FMA: contracting sequence correctness (detects compute bit-flip)
 *
 * Build:
 *   nvcc -O2 -o gpu_probe gpu_probe.cu
 *
 * Usage:
 *   ./gpu_probe                              # test all GPUs
 *   ./gpu_probe --gpus 0 1 2 3               # specific GPUs
 *   ./gpu_probe --timeout 10                 # per-test timeout in seconds
 *
 * Default: 3 rounds, 5 seconds timeout per test stage. Timeout/failure output
 * includes the failed test stage and final summary lists suspect GPUs.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <sys/wait.h>
#include <errno.h>
#include <math.h>
#include <time.h>
#include <fcntl.h>
#include <limits.h>

#include <cuda_runtime.h>

#define MAX_GPUS        16
#define DEFAULT_ROUNDS  3
#define DEFAULT_TIMEOUT 5
#define LOG_PREFIX      "gpu_probe"
#define LOG_EXT         ".log"

/* Stage status codes */
#define STATUS_PASS    0
#define STATUS_FAIL    1
#define STATUS_TIMEOUT 2
#define STATUS_SKIP    3

#define CUDA_CHECK(call) do {                                      \
    cudaError_t err = (call);                                      \
    if (err != cudaSuccess) {                                      \
        fprintf(stderr, "[PROBE] CUDA error: %s (%s:%d)\n",       \
                cudaGetErrorString(err), __FILE__, __LINE__);      \
        return -1;                                                 \
    }                                                              \
} while (0)

typedef int (*test_fn_t)(int);

struct StageRecord {
    int gpu_id;
    int round_id;
    int test_id;
    int status;  /* 0 pass, 1 fail, 2 timeout, 3 skip */
    double elapsed_ms;
};

struct GpuResult {
    int gpu_id;
    int bad;
    char reason[64];
    double total_ms;
    double stage_ms[5];
    int stage_count[5];
    int stage_status[5];
};

enum ProbeTestId {
    TEST_MEMORY = 1,
    TEST_WARP_SHUFFLE = 2,
    TEST_SFU = 3,
    TEST_FMA = 4,
    TEST_TOTAL = 5,
};

static const char *test_name(int test_id) {
    switch (test_id) {
        case TEST_MEMORY: return "memory";
        case TEST_WARP_SHUFFLE: return "warp_shuffle";
        case TEST_SFU: return "sfu";
        case TEST_FMA: return "fma";
        default: return "unknown";
    }
}

static double monotonic_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

/* ========== Signal handling globals (child process only) ========== */

static volatile sig_atomic_t current_test_id = 0;
static volatile sig_atomic_t current_round_id = 0;
static volatile sig_atomic_t current_gpu_id = -1;
static volatile sig_atomic_t current_timeout_sec = 0;
static double worker_start_ms = 0.0;  /* set once per worker for TEST_TOTAL on timeout */
static int result_fd = -1;

static void emit_stage_record(int gpu_id, int round_id, int test_id,
                              int status, double elapsed_ms) {
    if (result_fd < 0) return;
    struct StageRecord rec;
    rec.gpu_id = gpu_id;
    rec.round_id = round_id;
    rec.test_id = test_id;
    rec.status = status;
    rec.elapsed_ms = elapsed_ms;
    ssize_t unused = write(result_fd, &rec, sizeof(rec));
    (void)unused;
}

static void safe_signal_write(const char *msg, size_t len) {
    ssize_t unused = write(STDERR_FILENO, msg, len);
    (void)unused;
}

/*
 * SAFETY NOTE: This signal handler calls emit_stage_record() which uses write().
 * write() is async-signal-safe per POSIX. The struct construction uses only
 * stack-local and volatile sig_atomic_t globals. This is safe ONLY because
 * _exit() is called immediately after — no return to normal control flow.
 * Do NOT remove the _exit() without restructuring this handler.
 */
static void on_test_timeout(int sig) {
    (void)sig;
    if (current_test_id == TEST_MEMORY) {
        safe_signal_write("[PROBE] TIMEOUT test=memory\n", sizeof("[PROBE] TIMEOUT test=memory\n") - 1);
    } else if (current_test_id == TEST_WARP_SHUFFLE) {
        safe_signal_write("[PROBE] TIMEOUT test=warp_shuffle\n", sizeof("[PROBE] TIMEOUT test=warp_shuffle\n") - 1);
    } else if (current_test_id == TEST_SFU) {
        safe_signal_write("[PROBE] TIMEOUT test=sfu\n", sizeof("[PROBE] TIMEOUT test=sfu\n") - 1);
    } else if (current_test_id == TEST_FMA) {
        safe_signal_write("[PROBE] TIMEOUT test=fma\n", sizeof("[PROBE] TIMEOUT test=fma\n") - 1);
    } else {
        safe_signal_write("[PROBE] TIMEOUT test=unknown\n", sizeof("[PROBE] TIMEOUT test=unknown\n") - 1);
    }
    emit_stage_record((int)current_gpu_id, (int)current_round_id,
                      (int)current_test_id, STATUS_TIMEOUT,
                      (double)current_timeout_sec * 1000.0);
    /* Also emit TEST_TOTAL so parent gets real child elapsed time */
    emit_stage_record((int)current_gpu_id, 0, TEST_TOTAL, STATUS_FAIL,
                      monotonic_ms() - worker_start_ms);
    _exit(80 + (int)current_test_id);
}

static int run_test_with_timeout(int gpu_id, int round_id, int test_id,
                                 int timeout_sec, test_fn_t fn) {
    const char *name = test_name(test_id);
    printf("[PROBE] GPU %d round %d test=%s START timeout=%ds\n",
           gpu_id, round_id, name, timeout_sec);
    fflush(stdout);

    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = on_test_timeout;
    sigaction(SIGALRM, &sa, NULL);
    current_test_id = test_id;
    current_round_id = round_id;
    current_gpu_id = gpu_id;
    current_timeout_sec = timeout_sec;

    double t0 = monotonic_ms();
    alarm((unsigned)timeout_sec);
    int rc = fn(round_id);
    alarm(0);
    double elapsed = monotonic_ms() - t0;

    current_test_id = 0;
    current_round_id = 0;
    current_gpu_id = -1;
    current_timeout_sec = 0;

    if (rc == 2) {
        /* test returned SKIP */
        printf("[PROBE] GPU %d round %d test=%s SKIP elapsed_ms=%.3f\n",
               gpu_id, round_id, name, elapsed);
        fflush(stdout);
        emit_stage_record(gpu_id, round_id, test_id, STATUS_SKIP, elapsed);
        return 0;  /* skip is not a failure */
    }
    if (rc != 0) {
        fprintf(stderr, "[PROBE] GPU %d round %d test=%s FAIL elapsed_ms=%.3f\n",
                gpu_id, round_id, name, elapsed);
        fflush(stderr);
        emit_stage_record(gpu_id, round_id, test_id, STATUS_FAIL, elapsed);
        return 10 + test_id;
    }
    printf("[PROBE] GPU %d round %d test=%s PASS elapsed_ms=%.3f\n",
           gpu_id, round_id, name, elapsed);
    fflush(stdout);
    emit_stage_record(gpu_id, round_id, test_id, STATUS_PASS, elapsed);
    return 0;
}

/* ========== Kernels ========== */

__global__ void mem_pattern_write(unsigned int *buf, size_t count, unsigned int seed) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count)
        buf[idx] = (unsigned int)(idx ^ seed) * 2654435761u + seed;
}

__global__ void mem_pattern_verify(const unsigned int *buf, size_t count,
                                   unsigned int seed, unsigned int *errors) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        unsigned int expected = (unsigned int)(idx ^ seed) * 2654435761u + seed;
        if (buf[idx] != expected) atomicAdd(errors, 1u);
    }
}

__global__ void init_kernel(float *data, size_t count, unsigned int seed) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        unsigned int h = (unsigned int)(idx ^ seed) * 2654435761u + seed;
        data[idx] = ((float)(h & 0xFFFFu) / 65535.0f) - 0.5f;
    }
}

__device__ __forceinline__ float subwarp_reduce_max(float val, unsigned int mask) {
    val = fmaxf(val, __shfl_xor_sync(mask, val, 8));
    val = fmaxf(val, __shfl_xor_sync(mask, val, 4));
    val = fmaxf(val, __shfl_xor_sync(mask, val, 2));
    val = fmaxf(val, __shfl_xor_sync(mask, val, 1));
    return val;
}

__global__ void warp_shuffle_probe(const float *input, float *output,
                                   int num_rows, int num_cols, unsigned int *error_flag) {
    const int tidx = threadIdx.x, bidx = blockIdx.x;
    const int lane_id = tidx % 32, group_id = lane_id / 16;
    const int lane_in_group = lane_id % 16, warp_id = tidx / 32;
    const int scale_col = warp_id * 2 + group_id;
    if (scale_col >= (num_cols / 128) || bidx >= num_rows) return;

    float local_max = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int col = scale_col * 128 + lane_in_group * 8 + i;
        if (col < num_cols)
            local_max = fmaxf(local_max, fabsf(input[bidx * num_cols + col]));
    }

    unsigned int mask = (group_id == 0) ? 0x0000FFFFu : 0xFFFF0000u;
    float amax = subwarp_reduce_max(local_max, mask);
    float check = __shfl_sync(mask, amax, group_id * 16);
    if (amax != check) atomicExch(error_flag, 1u);

    float qscale = 448.0f / fmaxf(amax, 1e-8f);
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int col = scale_col * 128 + lane_in_group * 8 + i;
        if (col < num_cols)
            output[bidx * num_cols + col] = rintf(
                fminf(fmaxf(input[bidx * num_cols + col] * qscale, -448.0f), 448.0f));
    }
}

__global__ void sfu_test_kernel(const float *in, float *out, size_t count, unsigned int *errors) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        float x = in[idx];  /* in [-0.5, 0.5] */
        float e = expf(x);
        float r = rsqrtf(fabsf(x) + 1.0f);
        float l = logf(fabsf(x) + 1.0f);
        float t = tanhf(x);

        int bad = 0;
        if (isnan(e) || isnan(r) || isnan(l) || isnan(t)) bad = 1;
        if (isinf(r) || isinf(l) || isinf(t)) bad = 1;
        if (e < 0.5f || e > 1.7f) bad = 1;
        if (r < 0.7f || r > 1.1f) bad = 1;
        if (l < -0.01f || l > 0.45f) bad = 1;
        if (t < -0.5f || t > 0.5f) bad = 1;

        if (bad) atomicAdd(errors, 1u);
        out[idx] = e + r + l + t;  /* force writes to prevent optimizer removal */
    }
}

__global__ void fma_test_kernel(const float *in, float *out, size_t count, unsigned int *errors) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        float val = in[idx];
        #pragma unroll
        for (int i = 0; i < 32; i++)
            val = fmaf(val, 0.5f, 0.25f);  /* converges to 0.5 */
        out[idx] = val;
        if (isnan(val) || isinf(val) || fabsf(val - 0.5f) > 1e-5f)
            atomicAdd(errors, 1u);
    }
}

/* ========== Test buffer helpers (reduces duplication) ========== */

struct TestBuffers {
    float *d_in;
    float *d_out;
    unsigned int *d_err;
    int blocks;
    int threads;
};

static int alloc_test_buffers(struct TestBuffers *tb, size_t count) {
    size_t bytes = count * sizeof(float);
    tb->d_in = NULL;
    tb->d_out = NULL;
    tb->d_err = NULL;
    tb->threads = 256;
    tb->blocks = (int)((count + tb->threads - 1) / tb->threads);

    CUDA_CHECK(cudaMalloc(&tb->d_in, bytes));
    CUDA_CHECK(cudaMalloc(&tb->d_out, bytes));
    CUDA_CHECK(cudaMalloc(&tb->d_err, sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(tb->d_err, 0, sizeof(unsigned int)));
    return 0;
}

static void free_test_buffers(struct TestBuffers *tb) {
    if (tb->d_in)  cudaFree(tb->d_in);
    if (tb->d_out) cudaFree(tb->d_out);
    if (tb->d_err) cudaFree(tb->d_err);
    tb->d_in = NULL;
    tb->d_out = NULL;
    tb->d_err = NULL;
}

static int init_and_run_test(struct TestBuffers *tb, size_t count, unsigned int seed) {
    init_kernel<<<tb->blocks, tb->threads>>>(tb->d_in, count, seed);
    CUDA_CHECK(cudaDeviceSynchronize());
    return 0;
}

static int check_errors(struct TestBuffers *tb, unsigned int *h_err) {
    CUDA_CHECK(cudaMemcpy(h_err, tb->d_err, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    return 0;
}

/* ========== Tests ========== */

/*
 * test_memory: returns 0=PASS, 2=SKIP, -1=FAIL
 * Returns SKIP (not silent PASS) when free memory is insufficient.
 */
static int test_memory(int round_id) {
    size_t free_mem = 0, total_mem = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    const size_t reserve = 128ULL * 1024 * 1024;

    if (free_mem <= reserve) {
        fprintf(stderr, "[PROBE] WARNING: memory test SKIPPED — free_mem=%zu MB <= reserve=%zu MB "
                "(another process may be using the GPU)\n",
                free_mem / (1024 * 1024), reserve / (1024 * 1024));
        return 2;  /* SKIP — caller will emit STATUS_SKIP */
    }

    size_t test_bytes = (free_mem - reserve) & ~(size_t)0xFFF;
    size_t count = test_bytes / sizeof(unsigned int);
    unsigned int *d_buf = NULL, *d_err = NULL;
    unsigned int h_err = 0;

    CUDA_CHECK(cudaMalloc(&d_err, sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_buf, test_bytes));

    int thr = 256, blk = (int)((count + thr - 1) / thr);
    unsigned int seeds[] = { (unsigned int)(round_id * 7 + 42), 0xAAAAAAAAu, 0x55555555u };

    for (int s = 0; s < 3; s++) {
        CUDA_CHECK(cudaMemset(d_err, 0, sizeof(unsigned int)));
        mem_pattern_write<<<blk, thr>>>(d_buf, count, seeds[s]);
        CUDA_CHECK(cudaDeviceSynchronize());
        mem_pattern_verify<<<blk, thr>>>(d_buf, count, seeds[s], d_err);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&h_err, d_err, sizeof(unsigned int), cudaMemcpyDeviceToHost));
        if (h_err) {
            fprintf(stderr, "[PROBE] FAIL: memory seed=0x%08X errors=%u (round %d)\n",
                    seeds[s], h_err, round_id);
            cudaFree(d_buf);
            cudaFree(d_err);
            return -1;
        }
    }
    cudaFree(d_buf);
    cudaFree(d_err);
    printf("[PROBE] memory PASS (%zu MB, round %d)\n", test_bytes / (1024 * 1024), round_id);
    return 0;
}

static int test_warp_shuffle(int round_id) {
    const int R = 131072, C = 128;
    size_t elems = (size_t)R * C, bytes = elems * sizeof(float);
    float *d_in = NULL, *d_out = NULL;
    unsigned int *d_err = NULL;
    unsigned int h_err = 0;

    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMalloc(&d_err, sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(d_err, 0, sizeof(unsigned int)));

    int thr = 256, blk = (int)((elems + thr - 1) / thr);
    init_kernel<<<blk, thr>>>(d_in, elems, (unsigned int)(round_id * 43 + 17));
    CUDA_CHECK(cudaDeviceSynchronize());

    warp_shuffle_probe<<<R, 128>>>(d_in, d_out, R, C, d_err);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&h_err, d_err, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_err);

    if (h_err) {
        fprintf(stderr, "[PROBE] FAIL: warp shuffle (round %d)\n", round_id);
        return -1;
    }
    printf("[PROBE] warp shuffle PASS (round %d)\n", round_id);
    return 0;
}

static int test_sfu(int round_id) {
    const size_t N = 16 * 1024 * 1024;
    struct TestBuffers tb;
    unsigned int h_err = 0;

    if (alloc_test_buffers(&tb, N) != 0) return -1;
    if (init_and_run_test(&tb, N, (unsigned int)(round_id * 71 + 33)) != 0) {
        free_test_buffers(&tb);
        return -1;
    }

    sfu_test_kernel<<<tb.blocks, tb.threads>>>(tb.d_in, tb.d_out, N, tb.d_err);
    cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        fprintf(stderr, "[PROBE] CUDA error: %s (%s:%d)\n",
                cudaGetErrorString(sync_err), __FILE__, __LINE__);
        free_test_buffers(&tb);
        return -1;
    }

    if (check_errors(&tb, &h_err) != 0) { free_test_buffers(&tb); return -1; }
    free_test_buffers(&tb);

    if (h_err) {
        fprintf(stderr, "[PROBE] FAIL: SFU errors=%u (round %d)\n", h_err, round_id);
        return -1;
    }
    printf("[PROBE] SFU PASS (round %d)\n", round_id);
    return 0;
}

static int test_fma(int round_id) {
    const size_t N = 16 * 1024 * 1024;
    struct TestBuffers tb;
    unsigned int h_err = 0;

    if (alloc_test_buffers(&tb, N) != 0) return -1;
    if (init_and_run_test(&tb, N, (unsigned int)(round_id * 97 + 51)) != 0) {
        free_test_buffers(&tb);
        return -1;
    }

    fma_test_kernel<<<tb.blocks, tb.threads>>>(tb.d_in, tb.d_out, N, tb.d_err);
    cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        fprintf(stderr, "[PROBE] CUDA error: %s (%s:%d)\n",
                cudaGetErrorString(sync_err), __FILE__, __LINE__);
        free_test_buffers(&tb);
        return -1;
    }

    if (check_errors(&tb, &h_err) != 0) { free_test_buffers(&tb); return -1; }
    free_test_buffers(&tb);

    if (h_err) {
        fprintf(stderr, "[PROBE] FAIL: FMA errors=%u (round %d)\n", h_err, round_id);
        return -1;
    }
    printf("[PROBE] FMA PASS (round %d)\n", round_id);
    return 0;
}

/* ========== Worker ========== */

static int run_worker(int gpu_id, int rounds, int timeout) {
    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    /* Start timing AFTER cuda init (init time is driver-serialized, not our fault) */
    double ws = monotonic_ms();
    worker_start_ms = ws;  /* global — accessible from signal handler */
    printf("[PROBE] GPU %d: %s (%d SMs), rounds=%d, per_test_timeout=%ds\n",
           gpu_id, prop.name, prop.multiProcessorCount, rounds, timeout);
    fflush(stdout);

    for (int r = 1; r <= rounds; r++) {
        printf("[PROBE] round %d/%d\n", r, rounds);
        fflush(stdout);

        int rc = 0;
        rc = run_test_with_timeout(gpu_id, r, TEST_MEMORY, timeout, test_memory);
        if (rc != 0) {
            emit_stage_record(gpu_id, 0, TEST_TOTAL, STATUS_FAIL, monotonic_ms() - ws);
            return rc;
        }
        rc = run_test_with_timeout(gpu_id, r, TEST_WARP_SHUFFLE, timeout, test_warp_shuffle);
        if (rc != 0) {
            emit_stage_record(gpu_id, 0, TEST_TOTAL, STATUS_FAIL, monotonic_ms() - ws);
            return rc;
        }
        rc = run_test_with_timeout(gpu_id, r, TEST_SFU, timeout, test_sfu);
        if (rc != 0) {
            emit_stage_record(gpu_id, 0, TEST_TOTAL, STATUS_FAIL, monotonic_ms() - ws);
            return rc;
        }
        rc = run_test_with_timeout(gpu_id, r, TEST_FMA, timeout, test_fma);
        if (rc != 0) {
            emit_stage_record(gpu_id, 0, TEST_TOTAL, STATUS_FAIL, monotonic_ms() - ws);
            return rc;
        }
    }
    printf("[PROBE] GPU %d: ALL PASS\n", gpu_id);
    fflush(stdout);
    emit_stage_record(gpu_id, 0, TEST_TOTAL, STATUS_PASS, monotonic_ms() - ws);
    return 0;
}

/* ========== Scanner ========== */

static volatile sig_atomic_t timed_out = 0;
static void on_alarm(int sig) { (void)sig; timed_out = 1; }

static void describe_worker_status(int st, char *buf, size_t buflen) {
    if (WIFEXITED(st)) {
        int rc = WEXITSTATUS(st);
        if (rc == 0) {
            snprintf(buf, buflen, "pass");
        } else if (rc >= 81 && rc <= 84) {
            snprintf(buf, buflen, "timeout:%s", test_name(rc - 80));
        } else if (rc >= 11 && rc <= 14) {
            snprintf(buf, buflen, "fail:%s", test_name(rc - 10));
        } else {
            snprintf(buf, buflen, "fail:rc=%d", rc);
        }
    } else if (WIFSIGNALED(st)) {
        snprintf(buf, buflen, "signal:%d", WTERMSIG(st));
    } else {
        snprintf(buf, buflen, "unknown");
    }
}

static void init_gpu_result(struct GpuResult *r, int gpu_id) {
    memset(r, 0, sizeof(*r));
    r->gpu_id = gpu_id;
    snprintf(r->reason, sizeof(r->reason), "pass");
}

static void apply_stage_record(struct GpuResult *r, const struct StageRecord *rec) {
    if (rec->test_id == TEST_TOTAL) {
        r->total_ms = rec->elapsed_ms;
        return;
    }
    if (rec->test_id <= 0 || rec->test_id >= 5) return;
    r->stage_ms[rec->test_id] += rec->elapsed_ms;
    r->stage_count[rec->test_id] += 1;
    if (rec->status == STATUS_SKIP) {
        /* SKIP does not override existing status, just record it */
        if (r->stage_status[rec->test_id] == STATUS_PASS)
            r->stage_status[rec->test_id] = STATUS_SKIP;
        return;
    }
    if (rec->status != STATUS_PASS && r->stage_status[rec->test_id] == STATUS_PASS) {
        r->stage_status[rec->test_id] = rec->status;
        r->bad = 1;
        snprintf(r->reason, sizeof(r->reason), "%s:%s",
                 rec->status == STATUS_TIMEOUT ? "timeout" : "fail",
                 test_name(rec->test_id));
    }
}

static double stage_total_ms(const struct GpuResult *r, int test_id) {
    if (test_id <= 0 || test_id >= 5) return 0.0;
    return r->stage_ms[test_id];
}

static const char *stage_mark(const struct GpuResult *r, int test_id) {
    if (test_id <= 0 || test_id >= 5) return "-";
    if (r->stage_status[test_id] == STATUS_TIMEOUT) return "TIMEOUT";
    if (r->stage_status[test_id] == STATUS_FAIL) return "FAIL";
    if (r->stage_status[test_id] == STATUS_SKIP) return "SKIP";
    if (r->stage_count[test_id] > 0) return "PASS";
    return "-";
}

/* Per-GPU log file: gpu_probe_<gpu_id>.log */
static void redirect_child_output_to_log(int gpu_id) {
    char logpath[64];
    snprintf(logpath, sizeof(logpath), "%s_%d%s", LOG_PREFIX, gpu_id, LOG_EXT);
    int fd = open(logpath, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return;
    dup2(fd, STDOUT_FILENO);
    dup2(fd, STDERR_FILENO);
    if (fd > STDERR_FILENO) close(fd);
}

/* ---- spawn_workers: fork one child per GPU ---- */
static void spawn_workers(int *gpus, int n, int rounds, int timeout,
                          struct GpuResult *results, pid_t *pids,
                          int pipes[][2], double *gpu_start_ms) {
    for (int i = 0; i < n; i++) {
        int id = gpus[i];
        init_gpu_result(&results[i], id);
        pids[i] = -1;
        pipes[i][0] = -1;
        pipes[i][1] = -1;
        gpu_start_ms[i] = monotonic_ms();

        if (pipe(pipes[i]) != 0) {
            snprintf(results[i].reason, sizeof(results[i].reason), "pipe_failed");
            results[i].bad = 1;
            continue;
        }

        pid_t pid = fork();
        if (pid < 0) {
            perror("fork");
            close(pipes[i][0]);
            close(pipes[i][1]);
            pipes[i][0] = -1;
            pipes[i][1] = -1;
            snprintf(results[i].reason, sizeof(results[i].reason), "fork_failed");
            results[i].bad = 1;
            continue;
        }
        if (pid == 0) {
            /* Child process — close read end, set up environment */
            close(pipes[i][0]);
            result_fd = pipes[i][1];
            redirect_child_output_to_log(id);
            char v[8];
            snprintf(v, sizeof(v), "%d", id);
            setenv("CUDA_VISIBLE_DEVICES", v, 1);
            setenv("CUDA_LAUNCH_BLOCKING", "1", 1);
            _exit(run_worker(id, rounds, timeout));
        }
        pids[i] = pid;
        close(pipes[i][1]);
        pipes[i][1] = -1;
    }
}

/* ---- collect_results: reap children in completion order ---- */
static void collect_results(int n, int parent_timeout,
                            struct GpuResult *results, pid_t *pids,
                            int pipes[][2], double *gpu_start_ms) {
    int remaining = 0;
    int status_map[MAX_GPUS];      /* exit status per slot */
    int collected[MAX_GPUS];       /* whether slot has been reaped */

    for (int i = 0; i < n; i++) {
        status_map[i] = 0;
        collected[i] = (pids[i] <= 0) ? 1 : 0;
        if (pids[i] > 0) remaining++;
    }

    /* Set up parent-level alarm for overall timeout */
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = on_alarm;
    sigaction(SIGALRM, &sa, NULL);
    timed_out = 0;
    alarm((unsigned)parent_timeout);

    /* Reap children in whatever order they exit */
    while (remaining > 0 && !timed_out) {
        int st = 0;
        pid_t w = waitpid(-1, &st, 0);
        if (w <= 0) {
            if (errno == EINTR && timed_out) break;
            if (errno == ECHILD) break;
            continue;
        }
        /* Find which slot this pid belongs to */
        for (int i = 0; i < n; i++) {
            if (pids[i] == w) {
                status_map[i] = st;
                collected[i] = 1;
                remaining--;
                break;
            }
        }
    }
    alarm(0);

    /* Kill any stragglers that didn't exit in time */
    if (timed_out) {
        for (int i = 0; i < n; i++) {
            if (!collected[i] && pids[i] > 0) {
                kill(pids[i], SIGKILL);
                waitpid(pids[i], &status_map[i], 0);
                collected[i] = 1;
            }
        }
    }

    /* Now drain pipes and determine results for each GPU */
    for (int i = 0; i < n; i++) {
        /* Drain pipe records */
        struct StageRecord rec;
        while (pipes[i][0] >= 0 &&
               read(pipes[i][0], &rec, sizeof(rec)) == (ssize_t)sizeof(rec)) {
            apply_stage_record(&results[i], &rec);
        }
        if (pipes[i][0] >= 0) close(pipes[i][0]);

        /* Use child-reported total_ms if available (from TEST_TOTAL record) */
        /* Otherwise fall back to wall clock from fork to now */
        if (results[i].total_ms <= 0.0) {
            results[i].total_ms = monotonic_ms() - gpu_start_ms[i];
        }

        /* Determine final status */
        if (pids[i] <= 0) {
            /* Never forked — already marked bad in spawn_workers */
            continue;
        }
        if (timed_out && !WIFEXITED(status_map[i]) && !WIFSIGNALED(status_map[i])) {
            results[i].bad = 1;
            snprintf(results[i].reason, sizeof(results[i].reason), "scanner_timeout");
        } else if (WIFEXITED(status_map[i]) && WEXITSTATUS(status_map[i]) == 0) {
            results[i].bad = 0;
            snprintf(results[i].reason, sizeof(results[i].reason), "pass");
        } else {
            char reason[64];
            describe_worker_status(status_map[i], reason, sizeof(reason));
            results[i].bad = 1;
            if (!strcmp(results[i].reason, "pass")) {
                snprintf(results[i].reason, sizeof(results[i].reason), "%s", reason);
            }
        }
    }
}

/* ---- print_summary: output the results table ---- */
static void print_summary(struct GpuResult *results, int n, int rounds, int timeout,
                          double total_ms) {
    int nbad = 0, npass = 0;
    for (int i = 0; i < n; i++) {
        if (results[i].bad) nbad++;
        else npass++;
    }

    printf("\n===== GPU PROBE SUMMARY =====\n");
    printf("Log files: %s_<gpu_id>%s\n", LOG_PREFIX, LOG_EXT);
    printf("Config: rounds=%d per_test_timeout=%ds total_elapsed_ms=%.3f stage_ms=cumulative\n",
           rounds, timeout, total_ms);
    printf("+-----+--------+----------------------+-----------+--------+-----------+--------+-----------+--------+-----------+--------+-----------+\n");
    printf("| GPU | Result | Reason               | Mem ms    | Mem    | Warp ms   | Warp   | SFU ms    | SFU    | FMA ms    | FMA    | Total ms  |\n");
    printf("+-----+--------+----------------------+-----------+--------+-----------+--------+-----------+--------+-----------+--------+-----------+\n");
    for (int i = 0; i < n; i++) {
        printf("| %3d | %-6s | %-20s | %9.1f | %-6s | %9.1f | %-6s | %9.1f | %-6s | %9.1f | %-6s | %9.1f |\n",
               results[i].gpu_id,
               results[i].bad ? "BAD" : "PASS",
               results[i].reason,
               stage_total_ms(&results[i], TEST_MEMORY), stage_mark(&results[i], TEST_MEMORY),
               stage_total_ms(&results[i], TEST_WARP_SHUFFLE), stage_mark(&results[i], TEST_WARP_SHUFFLE),
               stage_total_ms(&results[i], TEST_SFU), stage_mark(&results[i], TEST_SFU),
               stage_total_ms(&results[i], TEST_FMA), stage_mark(&results[i], TEST_FMA),
               results[i].total_ms);
    }
    printf("+-----+--------+----------------------+-----------+--------+-----------+--------+-----------+--------+-----------+--------+-----------+\n");
    printf("PASS GPUs:");
    for (int i = 0; i < n; i++) if (!results[i].bad) printf(" %d", results[i].gpu_id);
    if (!npass) printf(" none");
    printf("\nBAD GPUs: ");
    for (int i = 0; i < n; i++) if (results[i].bad) printf(" %d(%s)", results[i].gpu_id, results[i].reason);
    if (!nbad) printf("none");
    printf("\n");
    if (nbad) {
        printf("CONCLUSION: suspect/problem GPUs:");
        for (int i = 0; i < n; i++) if (results[i].bad) printf(" %d", results[i].gpu_id);
        printf("\n");
    } else {
        printf("CONCLUSION: all tested GPUs passed.\n");
    }
}

static int scan_gpus(int *gpus, int n, int rounds, int timeout) {
    struct GpuResult results[MAX_GPUS];
    pid_t pids[MAX_GPUS];
    int pipes[MAX_GPUS][2];
    double gpu_start_ms[MAX_GPUS];
    double scan_start = monotonic_ms();

    printf("[PROBE] running GPU probe: gpus=%d rounds=%d per_test_timeout=%ds\n",
           n, rounds, timeout);
    fflush(stdout);

    spawn_workers(gpus, n, rounds, timeout, results, pids, pipes, gpu_start_ms);

    int parent_timeout = timeout * rounds * 4 + 30;
    collect_results(n, parent_timeout, results, pids, pipes, gpu_start_ms);

    double total_ms = monotonic_ms() - scan_start;
    print_summary(results, n, rounds, timeout, total_ms);

    for (int i = 0; i < n; i++) {
        if (results[i].bad) return 1;
    }
    return 0;
}

static int discover_gpus_from_dev(int *gpus, int max_gpus) {
    int n = 0;
    for (int i = 0; i < max_gpus; i++) {
        char path[64];
        snprintf(path, sizeof(path), "/dev/nvidia%d", i);
        if (access(path, F_OK) == 0) {
            gpus[n++] = i;
        }
    }
    return n;
}

/* ========== Argument parsing helpers ========== */

static int parse_int_arg(const char *str, int *out, int min_val, int max_val,
                         const char *arg_name) {
    char *endptr = NULL;
    errno = 0;
    long val = strtol(str, &endptr, 10);
    if (errno != 0 || endptr == str || *endptr != '\0') {
        fprintf(stderr, "[PROBE] error: invalid %s value '%s' — must be an integer\n",
                arg_name, str);
        return -1;
    }
    if (val < min_val || val > max_val) {
        fprintf(stderr, "[PROBE] error: %s value %ld out of range [%d, %d]\n",
                arg_name, val, min_val, max_val);
        return -1;
    }
    *out = (int)val;
    return 0;
}

/* ========== Main ========== */

int main(int argc, char **argv) {
    int gpus[MAX_GPUS], ngpus = 0;
    int rounds = DEFAULT_ROUNDS, timeout = DEFAULT_TIMEOUT;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--gpus")) {
            for (i++; i < argc && argv[i][0] != '-'; i++) {
                if (ngpus >= MAX_GPUS) {
                    fprintf(stderr, "[PROBE] error: too many GPUs specified (max %d)\n", MAX_GPUS);
                    return 1;
                }
                int gpu_id;
                if (parse_int_arg(argv[i], &gpu_id, 0, MAX_GPUS - 1, "--gpus") != 0)
                    return 1;
                gpus[ngpus++] = gpu_id;
            }
            i--;
        }
        else if (!strcmp(argv[i], "--rounds") && i + 1 < argc) {
            if (parse_int_arg(argv[++i], &rounds, 1, 100, "--rounds") != 0)
                return 1;
        }
        else if (!strcmp(argv[i], "--timeout") && i + 1 < argc) {
            if (parse_int_arg(argv[++i], &timeout, 1, 300, "--timeout") != 0)
                return 1;
        }
        else {
            fprintf(stderr,
                    "Usage: %s [--gpus ...] [--rounds N] [--timeout SEC]\n"
                    "Default: --rounds %d --timeout %d. Timeout is per test stage.\n",
                    argv[0], DEFAULT_ROUNDS, DEFAULT_TIMEOUT);
            return 1;
        }
    }
    if (!ngpus) {
        ngpus = discover_gpus_from_dev(gpus, MAX_GPUS);
        if (!ngpus) {
            fprintf(stderr, "[PROBE] no /dev/nvidia<N> devices found; use --gpus ... to specify GPUs\n");
            return 1;
        }
    }
    return scan_gpus(gpus, ngpus, rounds, timeout);
}
