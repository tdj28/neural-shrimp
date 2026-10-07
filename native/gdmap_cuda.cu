// Optional float64 CUDA backend for neural_shrimp.gdmaps.run_grid on TwoLayerNet.
// The CPU reference in src/neural_shrimp/gdmaps.py remains the default engine.
//
// One thread advances one learning-rate cell: full-batch gradient descent
// theta <- theta - rates * grad L(theta), with one tangent propagated by the
// exact Jacobian-vector product v <- v - rates * (H v) during the FTLE window.
//
// Arithmetic contract (default "mirror" build, compiled with --fmad=false):
// every floating-point operation of the CPU reference is reproduced with the
// same operands, association order and IEEE round-to-nearest rounding, through
// explicit intrinsics that the compiler may not contract or reorder.
//   * The two tensordot contractions use FMA chains, as the reference's
//     OpenBLAS dgemm kernels do: over the input dimension for pre-activations,
//     and over the samples for the first-layer gradient, split into the K
//     blocks of OpenBLAS's level-3 driver (block ends are passed in).
//   * einsum over hidden units follows numpy's two-lane SSE sum-of-products
//     loop (reverse four-pair blocks, separately rounded multiply and add).
//   * Reductions over samples are sequential; the ridge sum and tangent norm
//     use numpy's pairwise summation over the parameter index.
//   * All remaining arithmetic is elementwise, separately rounded.
// tanh and log come from the CUDA math library (at most one ulp from the true
// value) and can differ from numpy's implementations in the last bit.
// The "contract" build (GDMAP_FREE_CONTRACTION=1 with --fmad=true) replaces
// the intrinsics by plain operators so the compiler may fuse multiply-adds.
// It exists only to measure what that freedom costs in agreement and buys in
// speed; it is not a qualified configuration.
//
// Every thread computes sequentially with no atomics or inter-thread
// reductions, so results do not depend on launch geometry, chunking or
// scheduling, and repeated runs of the same binary are bit-identical.

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <new>

#ifndef GDMAP_FREE_CONTRACTION
#define GDMAP_FREE_CONTRACTION 0
#endif

#if GDMAP_FREE_CONTRACTION
#define MUL(a, b) ((a) * (b))
#define ADD(a, b) ((a) + (b))
#define SUB(a, b) ((a) - (b))
#define FMA(a, b, c) ((a) * (b) + (c))
#define DIV(a, b) ((a) / (b))
#define SQRT(a) sqrt(a)
#else
#define MUL(a, b) __dmul_rn((a), (b))
#define ADD(a, b) __dadd_rn((a), (b))
#define SUB(a, b) __dsub_rn((a), (b))
#define FMA(a, b, c) __fma_rn((a), (b), (c))
#define DIV(a, b) __ddiv_rn((a), (b))
#define SQRT(a) __dsqrt_rn(a)
#endif

namespace {

constexpr int D_MAX = 16;          // input dimension: one kernel per value 1..16
constexpr int W_MAX = 64;          // hidden width bound (runtime loops)
constexpr int KCHUNK_MAX = 16;     // OpenBLAS K blocks for the sample contraction
constexpr int SHARED_DOUBLES = 6144;  // x and y in 48 KiB of shared memory
constexpr int BLOCK = 128;
constexpr long long CANONICAL_NAN_BITS = 0x7ff8000000000000LL;  // numpy's np.nan

}  // namespace

extern "C" {

// All fields are 8 bytes wide so the ctypes mirror has no padding questions.
struct GdmapConfig {
    int64_t n, d, width, bias, tanh_act;
    int64_t capacity, tail_capacity, sd_max, tail_steps, device, kchunks;
    int64_t kchunk_end[KCHUNK_MAX];
    uint64_t max_bytes;
    double in_scale, out_scale, loss_scale, ridge, k_scale, half_ridge;
};

struct GdmapSchedule {
    int64_t ftle_start, ftle_log_start, tail_start, sd_max;
    double divergence_threshold;
};

struct GdmapInfo {
    char name[256];
    int64_t device, device_count, major, minor, multiprocessors;
    int64_t runtime_version, driver_version, l2_bytes, double_ratio, clock_khz;
    uint64_t total_bytes, free_bytes;
};

}  // extern "C"

namespace {

struct Net {
    int n, d, w, p, hidden, bias, tanh_act, kchunks;
    int off_b0, off_w1, off_b1;
    int kend[KCHUNK_MAX];
    double in_scale, out_scale, loss_scale, ridge, k_scale, half_ridge, n_real;
};

struct Schedule {
    int ftle_start, ftle_log_start, tail_start, sd_max;
    double threshold;
};

// Structure-of-arrays chunk state: element e of cell c lives at [e * stride + c],
// so consecutive threads touch consecutive addresses.
struct Buffers {
    size_t stride;
    double *theta, *tangent, *lr_h, *lr_o;
    int *alive, *ftle_ok;
    long long *diverged_at, *tail_count;
    double *log_growth, *min_loss, *last_loss, *tail_sum, *early;
    double *hbuf, *rbuf, *gbuf, *rgbuf, *kbuf;
    double *tail;          // [tail_steps][p][tail_capacity]
    size_t tail_stride;    // tail_capacity
    long long *period;     // device screening output, [tail_capacity]
    double *residual;
};

__device__ __forceinline__ double sq_at(const double* a, size_t stride, int i) {
    const double v = a[i * stride];
    return MUL(v, v);
}

// numpy's pairwise summation (PW_BLOCKSIZE 128, eight accumulators) applied to
// the squares of a strided vector, as in add.reduce(x * x) over a contiguous axis.
// Not inlined: each recursion level is one function rather than 2^level copies.
// Level 6 covers vectors of up to 128 * 2^6 = 8192 elements exactly.
template <int LEVEL>
__device__ __noinline__ double pairwise_squares(const double* a, size_t stride, int n) {
    if (n < 8) {
        double res = 0.0;
        for (int i = 0; i < n; ++i) res = ADD(res, sq_at(a, stride, i));
        return res;
    }
    if (n <= 128 || LEVEL >= 6) {
        double r0 = sq_at(a, stride, 0), r1 = sq_at(a, stride, 1);
        double r2 = sq_at(a, stride, 2), r3 = sq_at(a, stride, 3);
        double r4 = sq_at(a, stride, 4), r5 = sq_at(a, stride, 5);
        double r6 = sq_at(a, stride, 6), r7 = sq_at(a, stride, 7);
        int i = 8;
        for (; i < n - (n % 8); i += 8) {
            r0 = ADD(r0, sq_at(a, stride, i + 0));
            r1 = ADD(r1, sq_at(a, stride, i + 1));
            r2 = ADD(r2, sq_at(a, stride, i + 2));
            r3 = ADD(r3, sq_at(a, stride, i + 3));
            r4 = ADD(r4, sq_at(a, stride, i + 4));
            r5 = ADD(r5, sq_at(a, stride, i + 5));
            r6 = ADD(r6, sq_at(a, stride, i + 6));
            r7 = ADD(r7, sq_at(a, stride, i + 7));
        }
        double res = ADD(ADD(ADD(r0, r1), ADD(r2, r3)), ADD(ADD(r4, r5), ADD(r6, r7)));
        for (; i < n; ++i) res = ADD(res, sq_at(a, stride, i));
        return res;
    }
    int n2 = n / 2;
    n2 -= n2 % 8;
    return ADD(pairwise_squares<LEVEL + 1>(a, stride, n2),
               pairwise_squares<LEVEL + 1>(a + n2 * stride, stride, n - n2));
}

template <>
__device__ __noinline__ double pairwise_squares<7>(const double*, size_t, int) {
    return 0.0;  // unreachable: level 6 handles every remaining length
}

// Loss, gradient and (optionally) Hessian-vector product of one cell at its
// current state, mirroring TwoLayerNet.gradient and TwoLayerNet.hvp. Each
// parameter's (current value, gradient, tangent, HVP) is handed to the sink in a
// fixed order after its last use, so a sink may overwrite it in place.
template <int D, class Sink>
__device__ double evaluate_cell(const Net& P, const double* __restrict__ xs,
                                const double* __restrict__ ys, const Buffers& B, int c,
                                bool hvp, Sink& sink) {
    const int n = P.n, w = P.w;
    const size_t S = B.stride;
    double* th = B.theta + c;
    double* tg = B.tangent + c;
    double* hb = B.hbuf + c;
    double* rb = B.rbuf + c;
    double* gb = B.gbuf + c;
    double* rgb = B.rgbuf + c;
    const double ridge_sum = pairwise_squares<0>(th, S, P.p);

    // Pass 1a: activations h and pre-activation directional derivatives r_pre.
    for (int j = 0; j < w; ++j) {
        double W[D], V[D];
#pragma unroll
        for (int k = 0; k < D; ++k) W[k] = th[(size_t)(k * w + j) * S];
        const double b0 = P.bias ? th[(size_t)(P.off_b0 + j) * S] : 0.0;
        double c0 = 0.0;
        if (hvp) {
#pragma unroll
            for (int k = 0; k < D; ++k) V[k] = tg[(size_t)(k * w + j) * S];
            if (P.bias) c0 = tg[(size_t)(P.off_b0 + j) * S];
        }
        for (int i = 0; i < n; ++i) {
            const double* xi = xs + i * D;
            double acc = 0.0;
#pragma unroll
            for (int k = 0; k < D; ++k) acc = FMA(xi[k], W[k], acc);
            double pre = MUL(P.in_scale, acc);
            if (P.bias) pre = ADD(pre, b0);
            const size_t at = (size_t)(i * w + j) * S;
            hb[at] = P.tanh_act ? tanh(pre) : pre;
            if (hvp) {
                double racc = 0.0;
#pragma unroll
                for (int k = 0; k < D; ++k) racc = FMA(xi[k], V[k], racc);
                double rp = MUL(P.in_scale, racc);
                if (P.bias) rp = ADD(rp, c0);
                rb[at] = rp;
            }
        }
    }

    // Pass 1b: predictions, residuals, loss; numpy einsum lane order over j.
    const double b1 = P.bias ? th[(size_t)P.off_b1 * S] : 0.0;
    const double c1 = (P.bias && hvp) ? tg[(size_t)P.off_b1 * S] : 0.0;
    double sumsq = 0.0, gsum = 0.0, rgsum = 0.0;
    for (int i = 0; i < n; ++i) {
        const double* hrow = hb + (size_t)(i * w) * S;
        const double* rrow = rb + (size_t)(i * w) * S;
        double f0 = 0.0, f1 = 0.0, a0 = 0.0, a1 = 0.0, q0 = 0.0, q1 = 0.0;
        auto lane = [&](int j, double& fl, double& al, double& ql) {
            const double h = hrow[(size_t)j * S];
            const double w1 = th[(size_t)(P.off_w1 + j) * S];
            fl = ADD(MUL(h, w1), fl);
            if (hvp) {
                const double s = P.tanh_act ? SUB(1.0, MUL(h, h)) : 1.0;
                const double rh = MUL(s, rrow[(size_t)j * S]);
                const double v1 = tg[(size_t)(P.off_w1 + j) * S];
                al = ADD(MUL(rh, w1), al);
                ql = ADD(MUL(h, v1), ql);
            }
        };
        auto pad = [&](double& fl, double& al, double& ql) {
            const double z = MUL(0.0, 0.0);  // zero-filled tail lane
            fl = ADD(z, fl);
            if (hvp) {
                al = ADD(z, al);
                ql = ADD(z, ql);
            }
        };
        int j = 0;
        for (; j + 8 <= w; j += 8) {
#pragma unroll
            for (int blk = 3; blk >= 0; --blk) {
                lane(j + 2 * blk, f0, a0, q0);
                lane(j + 2 * blk + 1, f1, a1, q1);
            }
        }
        for (; j < w; j += 2) {
            lane(j, f0, a0, q0);
            if (j + 1 < w) lane(j + 1, f1, a1, q1);
            else pad(f1, a1, q1);
        }
        double f = MUL(P.out_scale, ADD(0.0, ADD(f0, f1)));
        if (P.bias) f = ADD(f, b1);
        const double r = SUB(f, ys[i]);
        sumsq = ADD(sumsq, MUL(r, r));
        const double g = MUL(P.k_scale, r);
        gb[(size_t)i * S] = g;
        gsum = ADD(gsum, g);
        if (hvp) {
            const double e2 = ADD(0.0, ADD(a0, a1));
            const double e3 = ADD(0.0, ADD(q0, q1));
            double rf = MUL(P.out_scale, ADD(e2, e3));
            if (P.bias) rf = ADD(rf, c1);
            const double rg = MUL(P.k_scale, rf);
            rgb[(size_t)i * S] = rg;
            rgsum = ADD(rgsum, rg);
        }
    }
    const double loss = ADD(MUL(P.loss_scale, DIV(sumsq, P.n_real)), MUL(P.half_ridge, ridge_sum));

    // Pass 2: per hidden unit, reductions over samples; then hand off parameters.
    double* kb = B.kbuf + c;
    for (int j = 0; j < w; ++j) {
        const double w1 = th[(size_t)(P.off_w1 + j) * S];
        const double a = MUL(P.out_scale, w1);
        const double v1 = hvp ? tg[(size_t)(P.off_w1 + j) * S] : 0.0;
        const double osv = MUL(P.out_scale, v1);
        double G[D], H[D];
#pragma unroll
        for (int k = 0; k < D; ++k) {
            G[k] = 0.0;
            H[k] = 0.0;
        }
        double gw1 = 0.0, gb0 = 0.0, hw1a = 0.0, hw1b = 0.0, hb0 = 0.0;
        int chunk = 0, chunk_end = P.kend[0];
        for (int i = 0; i < n; ++i) {
            const size_t at = (size_t)(i * w + j) * S;
            const double h = hb[at];
            const double g = gb[(size_t)i * S];
            double s, s2;
            if (P.tanh_act) {
                s = SUB(1.0, MUL(h, h));
                s2 = MUL(MUL(-2.0, h), s);
            } else {
                s = 1.0;
                s2 = 0.0;
            }
            gw1 = ADD(gw1, MUL(h, g));
            const double ga = MUL(g, a);
            const double delta = MUL(ga, s);
            const double* xi = xs + i * D;
#pragma unroll
            for (int k = 0; k < D; ++k) G[k] = FMA(xi[k], delta, G[k]);
            gb0 = ADD(gb0, delta);
            if (hvp) {
                const double rp = rb[at];
                const double rg = rgb[(size_t)i * S];
                const double rh = MUL(s, rp);
                hw1a = ADD(hw1a, MUL(rh, g));
                hw1b = ADD(hw1b, MUL(h, rg));
                const double t1 = MUL(MUL(rg, a), s);
                const double t2 = MUL(MUL(g, osv), s);
                const double t3 = MUL(MUL(ga, s2), rp);
                const double rd = ADD(ADD(t1, t2), t3);
#pragma unroll
                for (int k = 0; k < D; ++k) H[k] = FMA(xi[k], rd, H[k]);
                hb0 = ADD(hb0, rd);
            }
            if (P.kchunks > 1 && i + 1 == chunk_end) {
                // OpenBLAS adds each K block's accumulator into the zeroed output.
#pragma unroll
                for (int k = 0; k < D; ++k) {
                    double* gt = kb + (size_t)k * S;
                    *gt = chunk ? ADD(*gt, G[k]) : ADD(G[k], 0.0);
                    G[k] = 0.0;
                    if (hvp) {
                        double* ht = kb + (size_t)(D + k) * S;
                        *ht = chunk ? ADD(*ht, H[k]) : ADD(H[k], 0.0);
                        H[k] = 0.0;
                    }
                }
                ++chunk;
                chunk_end = chunk < P.kchunks ? P.kend[chunk] : n;
            }
        }
#pragma unroll
        for (int k = 0; k < D; ++k) {
            const int e = k * w + j;
            const double tv = th[(size_t)e * S];
            const double gk = P.kchunks > 1 ? kb[(size_t)k * S] : ADD(G[k], 0.0);
            const double grad = ADD(MUL(P.in_scale, gk), MUL(P.ridge, tv));
            double vv = 0.0, hv = 0.0;
            if (hvp) {
                const double hk = P.kchunks > 1 ? kb[(size_t)(D + k) * S] : ADD(H[k], 0.0);
                vv = tg[(size_t)e * S];
                hv = ADD(MUL(P.in_scale, hk), MUL(P.ridge, vv));
            }
            sink(e, true, tv, grad, vv, hv);
        }
        if (P.bias) {
            const int e = P.off_b0 + j;
            const double tv = th[(size_t)e * S];
            const double grad = ADD(gb0, MUL(P.ridge, tv));
            double vv = 0.0, hv = 0.0;
            if (hvp) {
                vv = tg[(size_t)e * S];
                hv = ADD(hb0, MUL(P.ridge, vv));
            }
            sink(e, true, tv, grad, vv, hv);
        }
        {
            const double grad = ADD(MUL(P.out_scale, gw1), MUL(P.ridge, w1));
            const double hv = hvp ? ADD(MUL(P.out_scale, ADD(hw1a, hw1b)), MUL(P.ridge, v1)) : 0.0;
            sink(P.off_w1 + j, false, w1, grad, v1, hv);
        }
    }
    if (P.bias) {
        const double grad = ADD(gsum, MUL(P.ridge, b1));
        const double hv = hvp ? ADD(rgsum, MUL(P.ridge, c1)) : 0.0;
        sink(P.off_b1, false, b1, grad, c1, hv);
    }
    return loss;
}

struct IterSink {
    double* th;
    double* tg;
    size_t S;
    double lr_h, lr_o, threshold;
    bool hvp, bad;
    double* tail;        // first parameter of this step's tail row, or nullptr
    size_t tail_stride;  // distance between parameters in the tail buffer
    __device__ void operator()(int e, bool hidden, double tv, double grad, double vv, double hv) {
        const double rate = hidden ? lr_h : lr_o;
        const double nv = SUB(tv, MUL(rate, grad));
        th[(size_t)e * S] = nv;
        if (!isfinite(nv) || fabs(nv) > threshold) bad = true;
        if (tail) tail[(size_t)e * tail_stride] = nv;
        if (hvp) tg[(size_t)e * S] = SUB(vv, MUL(rate, hv));
    }
};

struct EvalSink {
    double* grad;
    double* hv;
    size_t S;
    bool hvp;
    __device__ void operator()(int e, bool, double, double g, double, double h) {
        grad[(size_t)e * S] = g;
        if (hvp) hv[(size_t)e * S] = h;
    }
};

__device__ void load_shared(const Net& P, int d, const double* xg, const double* yg,
                            double* xs, double* ys) {
    for (int t = threadIdx.x; t < P.n * d; t += blockDim.x) xs[t] = xg[t];
    for (int t = threadIdx.x; t < P.n; t += blockDim.x) ys[t] = yg[t];
    __syncthreads();
}

// Advance cells [cell_begin, cell_begin + cell_count) from step_begin to
// step_end, mirroring the loop body of gdmaps.iterate_and_screen.
template <int D>
__global__ void __launch_bounds__(BLOCK) advance_kernel(Net P, Buffers B, const double* xg,
                                                        const double* yg, int cell_begin,
                                                        int cell_count, int step_begin,
                                                        int step_end, Schedule Q, int write_tail,
                                                        int tail_cell_begin) {
    extern __shared__ double shared[];
    double* xs = shared;
    double* ys = shared + P.n * D;
    load_shared(P, D, xg, yg, xs, ys);
    const int local = blockIdx.x * blockDim.x + threadIdx.x;
    if (local >= cell_count) return;
    const int c = cell_begin + local;
    if (!B.alive[c]) return;
    const size_t S = B.stride;
    int ftle_ok = B.ftle_ok[c];
    double log_growth = B.log_growth[c], min_loss = B.min_loss[c];
    double last_loss = B.last_loss[c], tail_sum = B.tail_sum[c];
    long long tail_count = B.tail_count[c];
    const double lr_h = B.lr_h[c], lr_o = B.lr_o[c];
    bool alive = true;
    long long diverged_at = -1;
    double* th = B.theta + c;
    double* tg = B.tangent + c;
    for (int step = step_begin; step < step_end; ++step) {
        const bool hvp = step >= Q.ftle_start && ftle_ok;
        double* tail = nullptr;
        if (write_tail && step >= Q.tail_start)
            tail = B.tail + ((size_t)(step - Q.tail_start) * P.p) * B.tail_stride + (c - tail_cell_begin);
        IterSink sink{th, tg, S, lr_h, lr_o, Q.threshold, hvp, false, tail, B.tail_stride};
        const double loss = evaluate_cell<D>(P, xs, ys, B, c, hvp, sink);
        const bool finite_loss = isfinite(loss);
        const bool bad = sink.bad || !finite_loss;
        if (step < Q.sd_max) B.early[(size_t)step * S + c] = loss;
        if (finite_loss) min_loss = min_loss < loss ? min_loss : loss;
        last_loss = loss;
        if (step >= Q.tail_start) {
            tail_sum = ADD(tail_sum, finite_loss ? loss : 0.0);
            tail_count += finite_loss ? 1 : 0;
        }
        if (hvp) {
            const double norm = SQRT(pairwise_squares<0>(tg, S, P.p));
            const bool ok = isfinite(norm) && norm > 0.0;
            if (!ok && !bad) ftle_ok = 0;
            if (ok && step >= Q.ftle_log_start) log_growth = ADD(log_growth, log(norm));
            if (ok)
                for (int e = 0; e < P.p; ++e) tg[(size_t)e * S] = DIV(tg[(size_t)e * S], norm);
        }
        if (bad) {
            const double nan = __longlong_as_double(CANONICAL_NAN_BITS);
            for (int e = 0; e < P.p; ++e) {
                th[(size_t)e * S] = nan;
                if (tail) tail[(size_t)e * B.tail_stride] = nan;
            }
            alive = false;
            diverged_at = step;
            break;
        }
    }
    B.ftle_ok[c] = ftle_ok;
    B.log_growth[c] = log_growth;
    B.min_loss[c] = min_loss;
    B.last_loss[c] = last_loss;
    B.tail_sum[c] = tail_sum;
    B.tail_count[c] = tail_count;
    if (!alive) {
        B.alive[c] = 0;
        B.diverged_at[c] = diverged_at;
    }
}

// Loss, gradient and HVP for a batch of cells, through the same evaluate_cell.
template <int D>
__global__ void __launch_bounds__(BLOCK) eval_kernel(Net P, Buffers B, const double* xg,
                                                     const double* yg, int cells, int with_hvp,
                                                     double* loss, double* grad, double* hv) {
    extern __shared__ double shared[];
    double* xs = shared;
    double* ys = shared + P.n * D;
    load_shared(P, D, xg, yg, xs, ys);
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= cells) return;
    EvalSink sink{grad + c, hv + c, B.stride, with_hvp != 0};
    loss[c] = evaluate_cell<D>(P, xs, ys, B, c, with_hvp != 0, sink);
}

// Full-state recurrence screen, a bitwise mirror of gdmaps.screen_tails for one
// cell per thread on the [steps][p][cells] tail buffer.
__global__ void screen_kernel(const double* tail, size_t tail_stride, int cells, int steps, int p,
                              int max_period, int min_repeats, double atol, double rtol,
                              long long* period_out, double* residual_out) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= cells) return;
    const double* T = tail + c;
    bool finite = true;
    for (size_t q = 0; q < (size_t)steps * p && finite; ++q) finite = isfinite(T[q * tail_stride]);
    long long period = 0;
    double residual = INFINITY;
    if (!finite) {
        period = -1;
    } else {
        for (int pp = 1; pp <= max_period; ++pp) {
            const int count = pp * min_repeats;
            if (count > steps) break;
            double worst = 0.0;
            for (int b = 0; b < min_repeats; ++b) {
                for (int r = 0; r < pp; ++r) {
                    const double* blk = T + (size_t)(steps - count + b * pp + r) * p * tail_stride;
                    const double* ref = T + (size_t)(steps - pp + r) * p * tail_stride;
                    for (int e = 0; e < p; ++e) {
                        const double av = blk[(size_t)e * tail_stride];
                        const double rv = ref[(size_t)e * tail_stride];
                        const double scale = ADD(atol, MUL(rtol, fmax(fabs(av), fabs(rv))));
                        const double ratio = DIV(fabs(SUB(av, rv)), scale);
                        worst = ratio > worst ? ratio : worst;
                    }
                }
            }
            if (worst <= 1.0) {
                period = pp;
                residual = worst;
                break;
            }
            residual = worst < residual ? worst : residual;
        }
    }
    period_out[c] = period;
    residual_out[c] = residual;
}

__global__ void init_kernel(Buffers B, int cells) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= cells) return;
    B.ftle_ok[c] = 1;
    B.log_growth[c] = 0.0;
    B.min_loss[c] = INFINITY;
    B.last_loss[c] = __longlong_as_double(CANONICAL_NAN_BITS);
    B.tail_sum[c] = 0.0;
    B.tail_count[c] = 0;
}

void set_error(char* out, size_t capacity, const char* what, cudaError_t code) {
    if (out && capacity)
        std::snprintf(out, capacity, "%s: %s (%d)", what, cudaGetErrorString(code), int(code));
}

void set_message(char* out, size_t capacity, const char* what) {
    if (out && capacity) std::snprintf(out, capacity, "%s", what);
}

bool make_net(const GdmapConfig& cfg, Net& P, char* error, size_t capacity) {
    if (cfg.n < 1 || cfg.d < 1 || cfg.d > D_MAX || cfg.width < 1 || cfg.width > W_MAX ||
        cfg.n * (cfg.d + 1) > SHARED_DOUBLES || cfg.kchunks < 1 || cfg.kchunks > KCHUNK_MAX ||
        cfg.kchunk_end[cfg.kchunks - 1] != cfg.n) {
        set_message(error, capacity, "Unsupported network size or K-block layout");
        return false;
    }
    std::memset(&P, 0, sizeof(P));
    P.n = int(cfg.n);
    P.d = int(cfg.d);
    P.w = int(cfg.width);
    P.bias = cfg.bias ? 1 : 0;
    P.tanh_act = cfg.tanh_act ? 1 : 0;
    P.p = P.d * P.w + P.w + P.bias * (P.w + 1);
    P.hidden = P.d * P.w + P.bias * P.w;
    P.off_b0 = P.d * P.w;
    P.off_w1 = P.d * P.w + P.bias * P.w;
    P.off_b1 = P.off_w1 + P.w;
    P.kchunks = int(cfg.kchunks);
    int previous = 0;
    for (int i = 0; i < P.kchunks; ++i) {
        P.kend[i] = int(cfg.kchunk_end[i]);
        if (P.kend[i] <= previous) {
            set_message(error, capacity, "K-block ends must increase");
            return false;
        }
        previous = P.kend[i];
    }
    P.in_scale = cfg.in_scale;
    P.out_scale = cfg.out_scale;
    P.loss_scale = cfg.loss_scale;
    P.ridge = cfg.ridge;
    P.k_scale = cfg.k_scale;
    P.half_ridge = cfg.half_ridge;
    P.n_real = double(cfg.n);
    return true;
}

template <int D>
cudaError_t launch_advance_d(const Net& P, const Buffers& B, const double* x, const double* y,
                             int begin, int count, int s0, int s1, const Schedule& Q, int write_tail,
                             int tail_begin) {
    const size_t shared = size_t(P.n) * (D + 1) * sizeof(double);
    advance_kernel<D><<<(count + BLOCK - 1) / BLOCK, BLOCK, shared>>>(P, B, x, y, begin, count, s0, s1,
                                                                     Q, write_tail, tail_begin);
    return cudaGetLastError();
}

template <int D>
cudaError_t launch_eval_d(const Net& P, const Buffers& B, const double* x, const double* y,
                          int cells, int with_hvp, double* loss, double* grad, double* hv) {
    const size_t shared = size_t(P.n) * (D + 1) * sizeof(double);
    eval_kernel<D><<<(cells + BLOCK - 1) / BLOCK, BLOCK, shared>>>(P, B, x, y, cells, with_hvp, loss,
                                                                  grad, hv);
    return cudaGetLastError();
}

#define GDMAP_DISPATCH(FN, ...)                       \
    switch (P.d) {                                    \
        case 1: return FN<1>(__VA_ARGS__);            \
        case 2: return FN<2>(__VA_ARGS__);            \
        case 3: return FN<3>(__VA_ARGS__);            \
        case 4: return FN<4>(__VA_ARGS__);            \
        case 5: return FN<5>(__VA_ARGS__);            \
        case 6: return FN<6>(__VA_ARGS__);            \
        case 7: return FN<7>(__VA_ARGS__);            \
        case 8: return FN<8>(__VA_ARGS__);            \
        case 9: return FN<9>(__VA_ARGS__);            \
        case 10: return FN<10>(__VA_ARGS__);          \
        case 11: return FN<11>(__VA_ARGS__);          \
        case 12: return FN<12>(__VA_ARGS__);          \
        case 13: return FN<13>(__VA_ARGS__);          \
        case 14: return FN<14>(__VA_ARGS__);          \
        case 15: return FN<15>(__VA_ARGS__);          \
        case 16: return FN<16>(__VA_ARGS__);          \
        default: return cudaErrorInvalidValue;        \
    }

cudaError_t launch_advance(const Net& P, const Buffers& B, const double* x, const double* y, int begin,
                           int count, int s0, int s1, const Schedule& Q, int write_tail,
                           int tail_begin) {
    GDMAP_DISPATCH(launch_advance_d, P, B, x, y, begin, count, s0, s1, Q, write_tail, tail_begin)
}

cudaError_t launch_eval(const Net& P, const Buffers& B, const double* x, const double* y, int cells,
                        int with_hvp, double* loss, double* grad, double* hv) {
    GDMAP_DISPATCH(launch_eval_d, P, B, x, y, cells, with_hvp, loss, grad, hv)
}

// Owned device allocations for one chunk of cells.
struct Context {
    GdmapConfig cfg;
    Net P;
    Buffers B;
    double *x = nullptr, *y = nullptr;
    void* blocks[32] = {};
    int nblocks = 0;
    uint64_t bytes = 0;
    cudaEvent_t start = nullptr, stop = nullptr;
};

template <class T>
bool allocate(Context& ctx, T*& pointer, size_t count, char* error, size_t capacity) {
    pointer = nullptr;
    if (count == 0) count = 1;
    const size_t bytes = count * sizeof(T);
    cudaError_t code = cudaMalloc((void**)&pointer, bytes);
    if (code != cudaSuccess) {
        set_error(error, capacity, "cudaMalloc", code);
        return false;
    }
    ctx.blocks[ctx.nblocks++] = pointer;
    ctx.bytes += bytes;
    return true;
}

void release(Context* ctx) {
    if (!ctx) return;
    // Free only this context's allocations and events; never reset the device.
    for (int i = 0; i < ctx->nblocks; ++i)
        if (ctx->blocks[i]) cudaFree(ctx->blocks[i]);
    if (ctx->start) cudaEventDestroy(ctx->start);
    if (ctx->stop) cudaEventDestroy(ctx->stop);
    delete ctx;
}

uint64_t planned_bytes(const GdmapConfig& cfg) {
    const uint64_t d = cfg.d, w = cfg.width, n = cfg.n;
    const uint64_t p = d * w + w + (cfg.bias ? w + 1 : 0);
    const uint64_t cap = cfg.capacity, tcap = cfg.tail_capacity;
    uint64_t doubles = 2 * p * cap + 6 * cap + cfg.sd_max * cap + (2 * n * w + 2 * n) * cap +
                       (cfg.kchunks > 1 ? 2 * D_MAX * cap : 0) + cfg.tail_steps * p * tcap + tcap +
                       n * d + n;
    uint64_t others = 2 * sizeof(int) * cap + 2 * sizeof(long long) * cap + sizeof(long long) * tcap;
    return doubles * sizeof(double) + others;
}

}  // namespace

extern "C" {

int gdmap_cuda_device_info(int device, GdmapInfo* out, char* error, size_t capacity) {
    if (!out) return -1;
    std::memset(out, 0, sizeof(*out));
    int count = 0;
    cudaError_t code = cudaGetDeviceCount(&count);
    out->device_count = count;
    if (code != cudaSuccess) {
        set_error(error, capacity, "cudaGetDeviceCount", code);
        return int(code);
    }
    if (device < 0 || device >= count) {
        set_message(error, capacity, "Invalid CUDA device index");
        return -1;
    }
    code = cudaSetDevice(device);
    if (code != cudaSuccess) {
        set_error(error, capacity, "cudaSetDevice", code);
        return int(code);
    }
    cudaDeviceProp prop{};
    code = cudaGetDeviceProperties(&prop, device);
    if (code != cudaSuccess) {
        set_error(error, capacity, "cudaGetDeviceProperties", code);
        return int(code);
    }
    size_t free_bytes = 0, total_bytes = 0;
    code = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (code != cudaSuccess) {
        set_error(error, capacity, "cudaMemGetInfo", code);
        return int(code);
    }
    int runtime = 0, driver = 0, ratio = 0;
    cudaRuntimeGetVersion(&runtime);
    cudaDriverGetVersion(&driver);
    cudaDeviceGetAttribute(&ratio, cudaDevAttrSingleToDoublePrecisionPerfRatio, device);
    std::snprintf(out->name, sizeof(out->name), "%s", prop.name);
    out->device = device;
    out->major = prop.major;
    out->minor = prop.minor;
    out->multiprocessors = prop.multiProcessorCount;
    out->runtime_version = runtime;
    out->driver_version = driver;
    out->l2_bytes = prop.l2CacheSize;
    out->double_ratio = ratio;
    out->clock_khz = prop.clockRate;
    out->total_bytes = total_bytes;
    out->free_bytes = free_bytes;
    return 0;
}

uint64_t gdmap_cuda_planned_bytes(const GdmapConfig* cfg) { return cfg ? planned_bytes(*cfg) : 0; }

int gdmap_cuda_create(const GdmapConfig* cfg, const double* x, const double* y, void** handle,
                      char* error, size_t capacity) {
    if (!cfg || !x || !y || !handle) return -1;
    *handle = nullptr;
    Net P;
    if (!make_net(*cfg, P, error, capacity)) return -1;
    if (cfg->capacity < 1 || cfg->tail_capacity < 1 || cfg->tail_capacity > cfg->capacity ||
        cfg->tail_steps < 1 || cfg->sd_max < 0 || cfg->capacity > (int64_t(1) << 30)) {
        set_message(error, capacity, "Invalid chunk capacities");
        return -1;
    }
    const uint64_t required = planned_bytes(*cfg);
    if (required > cfg->max_bytes) {
        set_message(error, capacity, "Chunk exceeds the configured device-byte ceiling");
        return -1;
    }
    cudaError_t code = cudaSetDevice(int(cfg->device));
    if (code != cudaSuccess) {
        set_error(error, capacity, "cudaSetDevice", code);
        return int(code);
    }
    size_t free_bytes = 0, total_bytes = 0;
    code = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (code != cudaSuccess) {
        set_error(error, capacity, "cudaMemGetInfo", code);
        return int(code);
    }
    if (required > free_bytes / 2) {
        set_message(error, capacity, "Chunk exceeds half of currently free VRAM");
        return -1;
    }
    Context* ctx = new (std::nothrow) Context();
    if (!ctx) return -1;
    ctx->cfg = *cfg;
    ctx->P = P;
    const size_t cap = size_t(cfg->capacity), tcap = size_t(cfg->tail_capacity);
    const size_t n = size_t(cfg->n), w = size_t(cfg->width), p = size_t(P.p);
    Buffers& B = ctx->B;
    std::memset(&B, 0, sizeof(B));
    B.stride = cap;
    B.tail_stride = tcap;
    bool ok = allocate(*ctx, B.theta, p * cap, error, capacity) &&
              allocate(*ctx, B.tangent, p * cap, error, capacity) &&
              allocate(*ctx, B.lr_h, cap, error, capacity) &&
              allocate(*ctx, B.lr_o, cap, error, capacity) &&
              allocate(*ctx, B.alive, cap, error, capacity) &&
              allocate(*ctx, B.ftle_ok, cap, error, capacity) &&
              allocate(*ctx, B.diverged_at, cap, error, capacity) &&
              allocate(*ctx, B.tail_count, cap, error, capacity) &&
              allocate(*ctx, B.log_growth, cap, error, capacity) &&
              allocate(*ctx, B.min_loss, cap, error, capacity) &&
              allocate(*ctx, B.last_loss, cap, error, capacity) &&
              allocate(*ctx, B.tail_sum, cap, error, capacity) &&
              allocate(*ctx, B.early, size_t(cfg->sd_max) * cap, error, capacity) &&
              allocate(*ctx, B.hbuf, n * w * cap, error, capacity) &&
              allocate(*ctx, B.rbuf, n * w * cap, error, capacity) &&
              allocate(*ctx, B.gbuf, n * cap, error, capacity) &&
              allocate(*ctx, B.rgbuf, n * cap, error, capacity) &&
              allocate(*ctx, B.kbuf, cfg->kchunks > 1 ? 2 * D_MAX * cap : 1, error, capacity) &&
              allocate(*ctx, B.tail, size_t(cfg->tail_steps) * p * tcap, error, capacity) &&
              allocate(*ctx, B.period, tcap, error, capacity) &&
              allocate(*ctx, B.residual, tcap, error, capacity) &&
              allocate(*ctx, ctx->x, n * size_t(cfg->d), error, capacity) &&
              allocate(*ctx, ctx->y, n, error, capacity);
    if (ok) {
        code = cudaMemcpy(ctx->x, x, n * size_t(cfg->d) * sizeof(double), cudaMemcpyHostToDevice);
        if (code == cudaSuccess) code = cudaMemcpy(ctx->y, y, n * sizeof(double), cudaMemcpyHostToDevice);
        if (code == cudaSuccess) code = cudaEventCreate(&ctx->start);
        if (code == cudaSuccess) code = cudaEventCreate(&ctx->stop);
        if (code != cudaSuccess) {
            set_error(error, capacity, "create", code);
            ok = false;
        }
    }
    if (!ok) {
        release(ctx);
        return -1;
    }
    *handle = ctx;
    return 0;
}

uint64_t gdmap_cuda_allocated_bytes(void* handle) {
    return handle ? static_cast<Context*>(handle)->bytes : 0;
}

int gdmap_cuda_destroy(void* handle) {
    release(static_cast<Context*>(handle));
    return 0;
}

// Initialize a chunk: theta (p x capacity, SoA), tangent (p x capacity, SoA),
// per-cell rates, alive flags and divergence steps. Other accumulators are reset.
int gdmap_cuda_upload(void* handle, int64_t cells, const double* theta, const double* tangent,
                      const double* lr_h, const double* lr_o, const int* alive,
                      const long long* diverged_at, char* error, size_t capacity) {
    Context* ctx = static_cast<Context*>(handle);
    if (!ctx || !theta || !tangent || !lr_h || !lr_o || !alive || !diverged_at || cells < 1 ||
        cells > ctx->cfg.capacity)
        return -1;
    const size_t cap = size_t(ctx->cfg.capacity), p = size_t(ctx->P.p);
    Buffers& B = ctx->B;
    cudaError_t code = cudaMemcpy(B.theta, theta, p * cap * sizeof(double), cudaMemcpyHostToDevice);
    if (code == cudaSuccess)
        code = cudaMemcpy(B.tangent, tangent, p * cap * sizeof(double), cudaMemcpyHostToDevice);
    if (code == cudaSuccess) code = cudaMemcpy(B.lr_h, lr_h, cap * sizeof(double), cudaMemcpyHostToDevice);
    if (code == cudaSuccess) code = cudaMemcpy(B.lr_o, lr_o, cap * sizeof(double), cudaMemcpyHostToDevice);
    if (code == cudaSuccess) code = cudaMemcpy(B.alive, alive, cap * sizeof(int), cudaMemcpyHostToDevice);
    if (code == cudaSuccess)
        code = cudaMemcpy(B.diverged_at, diverged_at, cap * sizeof(long long), cudaMemcpyHostToDevice);
    if (code == cudaSuccess && ctx->cfg.sd_max > 0)
        code = cudaMemset(B.early, 0xff, size_t(ctx->cfg.sd_max) * cap * sizeof(double));
    if (code == cudaSuccess) {
        init_kernel<<<(int(cap) + BLOCK - 1) / BLOCK, BLOCK>>>(B, int(cap));
        code = cudaGetLastError();
    }
    if (code == cudaSuccess) code = cudaDeviceSynchronize();
    if (code != cudaSuccess) {
        set_error(error, capacity, "upload", code);
        return int(code);
    }
    return 0;
}

int gdmap_cuda_reset_tail(void* handle, char* error, size_t capacity) {
    Context* ctx = static_cast<Context*>(handle);
    if (!ctx) return -1;
    const size_t count = size_t(ctx->cfg.tail_steps) * size_t(ctx->P.p) * size_t(ctx->cfg.tail_capacity);
    cudaError_t code = cudaMemset(ctx->B.tail, 0xff, count * sizeof(double));
    if (code != cudaSuccess) {
        set_error(error, capacity, "reset tail", code);
        return int(code);
    }
    return 0;
}

int gdmap_cuda_advance(void* handle, int64_t cell_begin, int64_t cell_count, int64_t step_begin,
                       int64_t step_end, const GdmapSchedule* schedule, int64_t write_tail,
                       int64_t tail_cell_begin, double* kernel_seconds, char* error,
                       size_t capacity) {
    Context* ctx = static_cast<Context*>(handle);
    if (!ctx || !schedule || cell_begin < 0 || cell_count < 1 ||
        cell_begin + cell_count > ctx->cfg.capacity || step_begin < 0 || step_end < step_begin ||
        step_end > (int64_t(1) << 30)) {
        set_message(error, capacity, "Invalid advance arguments");
        return -1;
    }
    if (write_tail && (tail_cell_begin < 0 || cell_begin < tail_cell_begin ||
                       cell_begin + cell_count > tail_cell_begin + ctx->cfg.tail_capacity ||
                       step_begin < schedule->tail_start ||
                       step_end > schedule->tail_start + ctx->cfg.tail_steps)) {
        set_message(error, capacity, "Tail write outside the tail buffer");
        return -1;
    }
    Schedule Q{int(schedule->ftle_start), int(schedule->ftle_log_start), int(schedule->tail_start),
               int(schedule->sd_max), schedule->divergence_threshold};
    if (Q.sd_max > ctx->cfg.sd_max) {
        set_message(error, capacity, "Score window exceeds the early-loss buffer");
        return -1;
    }
    cudaError_t code = cudaEventRecord(ctx->start);
    if (code == cudaSuccess)
        code = launch_advance(ctx->P, ctx->B, ctx->x, ctx->y, int(cell_begin), int(cell_count),
                              int(step_begin), int(step_end), Q, int(write_tail != 0),
                              int(tail_cell_begin));
    if (code == cudaSuccess) code = cudaEventRecord(ctx->stop);
    if (code == cudaSuccess) code = cudaEventSynchronize(ctx->stop);
    float ms = 0.0f;
    if (code == cudaSuccess) code = cudaEventElapsedTime(&ms, ctx->start, ctx->stop);
    if (code != cudaSuccess) {
        set_error(error, capacity, "advance", code);
        return int(code);
    }
    if (kernel_seconds) *kernel_seconds = double(ms) / 1000.0;
    return 0;
}

int gdmap_cuda_take_tail(void* handle, double* out, char* error, size_t capacity) {
    Context* ctx = static_cast<Context*>(handle);
    if (!ctx || !out) return -1;
    const size_t count = size_t(ctx->cfg.tail_steps) * size_t(ctx->P.p) * size_t(ctx->cfg.tail_capacity);
    cudaError_t code = cudaMemcpy(out, ctx->B.tail, count * sizeof(double), cudaMemcpyDeviceToHost);
    if (code != cudaSuccess) {
        set_error(error, capacity, "copy tail", code);
        return int(code);
    }
    return 0;
}

int gdmap_cuda_screen(void* handle, int64_t cells, int64_t max_period, int64_t min_repeats,
                      double atol, double rtol, long long* period, double* residual,
                      double* kernel_seconds, char* error, size_t capacity) {
    Context* ctx = static_cast<Context*>(handle);
    if (!ctx || !period || !residual || cells < 1 || cells > ctx->cfg.tail_capacity ||
        max_period < 1 || min_repeats < 2 || !(atol > 0) || !(rtol >= 0)) {
        set_message(error, capacity, "Invalid screening arguments");
        return -1;
    }
    cudaError_t code = cudaEventRecord(ctx->start);
    if (code == cudaSuccess) {
        screen_kernel<<<(int(cells) + BLOCK - 1) / BLOCK, BLOCK>>>(
            ctx->B.tail, ctx->B.tail_stride, int(cells), int(ctx->cfg.tail_steps), ctx->P.p,
            int(max_period), int(min_repeats), atol, rtol, ctx->B.period, ctx->B.residual);
        code = cudaGetLastError();
    }
    if (code == cudaSuccess) code = cudaEventRecord(ctx->stop);
    if (code == cudaSuccess) code = cudaEventSynchronize(ctx->stop);
    float ms = 0.0f;
    if (code == cudaSuccess) code = cudaEventElapsedTime(&ms, ctx->start, ctx->stop);
    if (code == cudaSuccess)
        code = cudaMemcpy(period, ctx->B.period, size_t(cells) * sizeof(long long), cudaMemcpyDeviceToHost);
    if (code == cudaSuccess)
        code = cudaMemcpy(residual, ctx->B.residual, size_t(cells) * sizeof(double), cudaMemcpyDeviceToHost);
    if (code != cudaSuccess) {
        set_error(error, capacity, "screen", code);
        return int(code);
    }
    if (kernel_seconds) *kernel_seconds = double(ms) / 1000.0;
    return 0;
}

// Copy chunk state back: theta (p x capacity SoA), per-cell arrays (capacity)
// and early losses (sd_max x capacity).
int gdmap_cuda_fetch(void* handle, double* theta, int* alive, long long* diverged_at, int* ftle_ok,
                     double* log_growth, double* min_loss, double* last_loss, double* tail_sum,
                     long long* tail_count, double* early, char* error, size_t capacity) {
    Context* ctx = static_cast<Context*>(handle);
    if (!ctx || !theta || !alive || !diverged_at || !ftle_ok || !log_growth || !min_loss ||
        !last_loss || !tail_sum || !tail_count || (ctx->cfg.sd_max > 0 && !early))
        return -1;
    const size_t cap = size_t(ctx->cfg.capacity), p = size_t(ctx->P.p);
    const Buffers& B = ctx->B;
    cudaError_t code = cudaMemcpy(theta, B.theta, p * cap * sizeof(double), cudaMemcpyDeviceToHost);
    struct Item {
        void* host;
        const void* device;
        size_t bytes;
    } items[] = {
        {alive, B.alive, cap * sizeof(int)},
        {diverged_at, B.diverged_at, cap * sizeof(long long)},
        {ftle_ok, B.ftle_ok, cap * sizeof(int)},
        {log_growth, B.log_growth, cap * sizeof(double)},
        {min_loss, B.min_loss, cap * sizeof(double)},
        {last_loss, B.last_loss, cap * sizeof(double)},
        {tail_sum, B.tail_sum, cap * sizeof(double)},
        {tail_count, B.tail_count, cap * sizeof(long long)},
        {early, B.early, size_t(ctx->cfg.sd_max) * cap * sizeof(double)},
    };
    for (const Item& item : items) {
        if (code != cudaSuccess) break;
        if (item.bytes) code = cudaMemcpy(item.host, item.device, item.bytes, cudaMemcpyDeviceToHost);
    }
    if (code != cudaSuccess) {
        set_error(error, capacity, "fetch", code);
        return int(code);
    }
    return 0;
}

// Standalone single evaluation for qualification: theta and vector are SoA
// (p x cells); outputs loss (cells), grad and hvp (p x cells).
int gdmap_cuda_evaluate(const GdmapConfig* cfg, const double* x, const double* y, int64_t cells,
                        const double* theta, const double* vector, double* loss, double* grad,
                        double* hvp, char* error, size_t capacity) {
    if (!cfg || !x || !y || !theta || !loss || !grad || (vector && !hvp) || cells < 1 ||
        cells > (int64_t(1) << 24))
        return -1;
    GdmapConfig local = *cfg;
    local.capacity = cells;
    local.tail_capacity = 1;
    local.tail_steps = 1;
    local.sd_max = 0;
    void* handle = nullptr;
    int result = gdmap_cuda_create(&local, x, y, &handle, error, capacity);
    if (result) return result;
    Context* ctx = static_cast<Context*>(handle);
    const size_t p = size_t(ctx->P.p), count = size_t(cells);
    double *dloss = nullptr, *dgrad = nullptr, *dhvp = nullptr;
    bool ok = allocate(*ctx, dloss, count, error, capacity) &&
              allocate(*ctx, dgrad, p * count, error, capacity) &&
              allocate(*ctx, dhvp, p * count, error, capacity);
    cudaError_t code = cudaSuccess;
    if (ok) {
        code = cudaMemcpy(ctx->B.theta, theta, p * count * sizeof(double), cudaMemcpyHostToDevice);
        if (code == cudaSuccess && vector)
            code = cudaMemcpy(ctx->B.tangent, vector, p * count * sizeof(double), cudaMemcpyHostToDevice);
        if (code == cudaSuccess)
            code = launch_eval(ctx->P, ctx->B, ctx->x, ctx->y, int(cells), vector ? 1 : 0, dloss, dgrad, dhvp);
        if (code == cudaSuccess) code = cudaDeviceSynchronize();
        if (code == cudaSuccess) code = cudaMemcpy(loss, dloss, count * sizeof(double), cudaMemcpyDeviceToHost);
        if (code == cudaSuccess) code = cudaMemcpy(grad, dgrad, p * count * sizeof(double), cudaMemcpyDeviceToHost);
        if (code == cudaSuccess && vector)
            code = cudaMemcpy(hvp, dhvp, p * count * sizeof(double), cudaMemcpyDeviceToHost);
        if (code != cudaSuccess) set_error(error, capacity, "evaluate", code);
    }
    release(ctx);
    if (!ok) return -1;
    return code == cudaSuccess ? 0 : int(code);
}

}  // extern "C"
