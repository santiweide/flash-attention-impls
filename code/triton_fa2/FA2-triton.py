# FA2_triton_bench.py
import math
import contextlib
import torch
import triton
import triton.language as tl

# ========= Test Spec =========
NUM_HEADS = 16
HIDDEN_SIZE = 512
HEAD_DIM = HIDDEN_SIZE // NUM_HEADS
SEQLEN = 1024
BATCH = 1
DTYPE = torch.float16
CAUSAL = True

# ========= Kernel Tiles (safe defaults) =========
BLOCK_M = 128        # query rows per block
BLOCK_N = 128        # key cols per block
BLOCK_D = 64        # must divide HEAD_DIM; <= 128

# ======================================
# Forward kernel (online softmax)
# ======================================
@triton.jit
def _fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    M_ptr, L_ptr,
    B, H, N_CTX, D_HEAD,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    stride_mb, stride_mh, stride_mn,
    stride_lb, stride_lh, stride_ln,
    softmax_scale,
    is_causal: tl.constexpr,
    BLOCK_M_: tl.constexpr, BLOCK_N_: tl.constexpr, BLOCK_D_: tl.constexpr
):
    bh = tl.program_id(0)
    rm = tl.program_id(1)
    b = bh // H
    h = bh % H
    row_start = rm * BLOCK_M_

    offs_m_1d = row_start + tl.arange(0, BLOCK_M_)
    mask_m_1d = offs_m_1d < N_CTX
    offs_m_2d = offs_m_1d[:, None]
    offs_d = tl.arange(0, BLOCK_D_)[None, :]

    # Load Q (M, D)
    Q_tile_ptr = Q_ptr + b*stride_qb + h*stride_qh + offs_m_2d*stride_qn + offs_d*stride_qd
    q = tl.load(Q_tile_ptr, mask=(offs_m_2d < N_CTX) & (offs_d < D_HEAD), other=0.0)

    # Online softmax stats
    m_i = tl.full((BLOCK_M_,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M_,), tl.float32)
    acc = tl.zeros((BLOCK_M_, BLOCK_D_), tl.float32)

    for col_start in range(0, N_CTX, BLOCK_N_):
        offs_n = col_start + tl.arange(0, BLOCK_N_)[:, None]  # (N,1)

        K_tile_ptr = K_ptr + b*stride_kb + h*stride_kh + offs_n*stride_kn + offs_d*stride_kd
        V_tile_ptr = V_ptr + b*stride_vb + h*stride_vh + offs_n*stride_vn + offs_d*stride_vd
        k = tl.load(K_tile_ptr, mask=(offs_n < N_CTX) & (offs_d < D_HEAD), other=0.0)
        v = tl.load(V_tile_ptr, mask=(offs_n < N_CTX) & (offs_d < D_HEAD), other=0.0)

        qk = tl.dot(q, tl.trans(k)) * softmax_scale  # (M,N)

        if is_causal:
            row_idx = row_start + tl.arange(0, BLOCK_M_)[:, None]
            col_idx = col_start + tl.arange(0, BLOCK_N_)[None, :]
            qk = tl.where(col_idx > row_idx, -float("inf"), qk)

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = l_i * tl.exp(m_i - m_ij) + tl.sum(p, axis=1)

        pv = tl.dot(p.to(tl.float32), v.to(tl.float32))  # (M,D)
        alpha = (l_i * tl.exp(m_i - m_ij)) / l_ij
        beta = 1.0 / l_ij
        acc = acc * alpha[:, None] + pv * beta[:, None]

        m_i = m_ij
        l_i = l_ij

    O_tile_ptr = O_ptr + b*stride_ob + h*stride_oh + offs_m_2d*stride_on + offs_d*stride_od
    tl.store(O_tile_ptr, acc.to(tl.float16), mask=(offs_m_2d < N_CTX) & (offs_d < D_HEAD))

    M_tile_ptr = M_ptr + b*stride_mb + h*stride_mh + offs_m_1d*stride_mn
    L_tile_ptr = L_ptr + b*stride_lb + h*stride_lh + offs_m_1d*stride_ln
    tl.store(M_tile_ptr, m_i, mask=mask_m_1d)
    tl.store(L_tile_ptr, l_i, mask=mask_m_1d)

# ======================================
# Backward kernel (recompute softmax)
# ======================================
@triton.jit
def _bwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    dO_ptr, dQ_ptr, dK_ptr, dV_ptr,
    M_ptr, L_ptr,
    B, H, N_CTX, D_HEAD,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    stride_dqb, stride_dqh, stride_dqn, stride_dqd,
    stride_dkb, stride_dkh, stride_dkn, stride_dkd,
    stride_dvb, stride_dvh, stride_dvn, stride_dvd,
    stride_mb, stride_mh, stride_mn,
    stride_lb, stride_lh, stride_ln,
    softmax_scale,
    is_causal: tl.constexpr,
    BLOCK_M_: tl.constexpr, BLOCK_N_: tl.constexpr, BLOCK_D_: tl.constexpr
):
    bh = tl.program_id(0)
    rm = tl.program_id(1)
    b = bh // H
    h = bh % H
    row_start = rm * BLOCK_M_

    offs_m_1d = row_start + tl.arange(0, BLOCK_M_)
    mask_m_1d = offs_m_1d < N_CTX
    offs_m_2d = offs_m_1d[:, None]
    offs_d = tl.arange(0, BLOCK_D_)[None, :]

    Q_tile_ptr = Q_ptr + b*stride_qb + h*stride_qh + offs_m_2d*stride_qn + offs_d*stride_qd
    q = tl.load(Q_tile_ptr, mask=(offs_m_2d < N_CTX) & (offs_d < D_HEAD), other=0.0).to(tl.float32)
    dO_tile_ptr = dO_ptr + b*stride_ob + h*stride_oh + offs_m_2d*stride_on + offs_d*stride_od
    dO = tl.load(dO_tile_ptr, mask=(offs_m_2d < N_CTX) & (offs_d < D_HEAD), other=0.0).to(tl.float32)

    M_tile_ptr = M_ptr + b*stride_mb + h*stride_mh + offs_m_1d*stride_mn
    L_tile_ptr = L_ptr + b*stride_lb + h*stride_lh + offs_m_1d*stride_ln
    m_i = tl.load(M_tile_ptr, mask=mask_m_1d, other=-float("inf"))
    l_i = tl.load(L_tile_ptr, mask=mask_m_1d, other=0.0)

    dQ_acc = tl.zeros((BLOCK_M_, BLOCK_D_), tl.float32)
    offs_n_base = tl.arange(0, BLOCK_N_)[:, None]

    for col_start in range(0, N_CTX, BLOCK_N_):
        offs_n = col_start + offs_n_base
        K_tile_ptr = K_ptr + b*stride_kb + h*stride_kh + offs_n*stride_kn + offs_d*stride_kd
        V_tile_ptr = V_ptr + b*stride_vb + h*stride_vh + offs_n*stride_vn + offs_d*stride_vd
        k = tl.load(K_tile_ptr, mask=(offs_n < N_CTX) & (offs_d < D_HEAD), other=0.0).to(tl.float32)
        v = tl.load(V_tile_ptr, mask=(offs_n < N_CTX) & (offs_d < D_HEAD), other=0.0).to(tl.float32)

        qk = tl.dot(q, tl.trans(k)) * softmax_scale
        if is_causal:
            row_idx = row_start + tl.arange(0, BLOCK_M_)[:, None]
            col_idx = col_start + tl.arange(0, BLOCK_N_)[None, :]
            qk = tl.where(col_idx > row_idx, -float("inf"), qk)

        p = tl.exp(qk - m_i[:, None]) / l_i[:, None]

        dv_local = tl.dot(tl.trans(p), dO)          # (N,D)
        dP = tl.dot(dO, tl.trans(v))                # (M,N)
        dp_sum = tl.sum(dP * p, axis=1)             # (M,)
        dS = (dP - dp_sum[:, None] * p) * softmax_scale

        dQ_acc += tl.dot(dS, k)                     # (M,D)
        dK_local = tl.dot(tl.trans(dS), q)          # (N,D)

        dV_tile_ptr = dV_ptr + b*stride_dvb + h*stride_dvh + offs_n*stride_dvn + offs_d*stride_dvd
        tl.atomic_add(dV_tile_ptr, dv_local.to(tl.float16), mask=(offs_n < N_CTX) & (offs_d < D_HEAD))
        dK_tile_ptr = dK_ptr + b*stride_dkb + h*stride_dkh + offs_n*stride_dkn + offs_d*stride_dkd
        tl.atomic_add(dK_tile_ptr, dK_local.to(tl.float16), mask=(offs_n < N_CTX) & (offs_d < D_HEAD))

    dQ_tile_ptr = dQ_ptr + b*stride_dqb + h*stride_dqh + offs_m_2d*stride_dqn + offs_d*stride_dqd
    tl.store(dQ_tile_ptr, dQ_acc.to(tl.float16), mask=(offs_m_2d < N_CTX) & (offs_d < D_HEAD))


class _FlashAttnFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal: bool):
        assert q.is_cuda and k.is_cuda and v.is_cuda
        B, H, N, D = q.shape
        assert D % 16 == 0 and D <= 128
        o = torch.empty_like(q, dtype=q.dtype)

        m = torch.empty((B, H, N), dtype=torch.float32, device=q.device)
        l = torch.empty((B, H, N), dtype=torch.float32, device=q.device)
        softmax_scale = 1.0 / math.sqrt(D)

        grid = (B * H, triton.cdiv(N, BLOCK_M))
        torch.cuda.nvtx.range_push("FA2_FWD")
        _fwd_kernel[grid](
            q, k, v, o, m, l,
            B, H, N, D,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            m.stride(0), m.stride(1), m.stride(2),
            l.stride(0), l.stride(1), l.stride(2),
            softmax_scale,
            is_causal=causal,
            BLOCK_M_=BLOCK_M, BLOCK_N_=BLOCK_N, BLOCK_D_=min(BLOCK_D, D),
            num_warps=4, num_stages=2,  # forward: double-buffer
        )
        torch.cuda.nvtx.range_pop()

        ctx.save_for_backward(q, k, v, m, l)
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, m, l = ctx.saved_tensors
        B, H, N, D = q.shape

        dQ = torch.zeros_like(q)
        dK = torch.zeros_like(k)
        dV = torch.zeros_like(v)
        softmax_scale = 1.0 / math.sqrt(D)

        grid = (B * H, triton.cdiv(N, BLOCK_M))
        torch.cuda.nvtx.range_push("FA2_BWD")
        _bwd_kernel[grid](
            q, k, v, do, dQ, dK, dV, m, l,
            B, H, N, D,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
            dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
            dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
            m.stride(0), m.stride(1), m.stride(2),
            l.stride(0), l.stride(1), l.stride(2),
            softmax_scale,
            is_causal=ctx.causal,
            BLOCK_M_=BLOCK_M, BLOCK_N_=BLOCK_N, BLOCK_D_=min(BLOCK_D, D),
            num_warps=4, num_stages=1,  # backward: single-buffer to save smem
        )
        torch.cuda.nvtx.range_pop()
        return dQ, dK, dV, None


def flash_attention(q, k, v, causal=False):
    orig_dtype = q.dtype
    if q.dtype == torch.float32:
        q = q.half(); k = k.half(); v = v.half()
    return _FlashAttnFn.apply(q, k, v, causal).to(orig_dtype)

# =========================
# Bench helpers
# =========================
def measure_latency(func, warmup=10, iters=100):
    # Warmup
    for _ in range(warmup):
        out = func(); torch.cuda.synchronize()
    # Measure
    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        out = func()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))  # ms
    import statistics as stats
    return {
        "mean_ms": stats.mean(times),
        "std_ms": stats.pstdev(times),
        "iters": iters,
    }

def try_max_batch(base_B=1, H=NUM_HEADS, N=SEQLEN, D=HEAD_DIM, dtype=DTYPE, causal=CAUSAL):
    device = "cuda"
    low, high = 1, base_B
    # grow until OOM
    while True:
        try:
            q = torch.randn(high, H, N, D, device=device, dtype=dtype, requires_grad=True)
            k = torch.randn_like(q)
            v = torch.randn_like(q)
            o = flash_attention(q, k, v, causal=causal)
            loss = o.float().pow(2).mean(); loss.backward()
            del q, k, v, o, loss
            torch.cuda.empty_cache()
            low = high
            high *= 2
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                break
            else:
                raise
    # binary search in (low, high)
    L, R = low, high - 1
    best = L
    while L <= R:
        mid = (L + R) // 2
        try:
            q = torch.randn(mid, H, N, D, device=device, dtype=dtype, requires_grad=True)
            k = torch.randn_like(q); v = torch.randn_like(q)
            o = flash_attention(q, k, v, causal=causal)
            loss = o.float().pow(2).mean(); loss.backward()
            del q, k, v, o, loss
            torch.cuda.empty_cache()
            best = mid
            L = mid + 1
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                R = mid - 1
            else:
                raise
    return best

def sdpa_reference(q, k, v, causal=False):
    # q,k,v: (B,H,N,D) -> (B*H, N, D)
    B, H, N, D = q.shape
    scale = 1.0 / math.sqrt(D)
    q2 = q.reshape(B*H, N, D).to(torch.float32)
    k2 = k.reshape(B*H, N, D).to(torch.float32)
    v2 = v.reshape(B*H, N, D).to(torch.float32)
    # PyTorch SDPA expects (B*H, N, D)
    attn_mask = None
    out = torch.nn.functional.scaled_dot_product_attention(
        q2, k2, v2, attn_mask=attn_mask, is_causal=causal, scale=scale
    )
    return out.reshape(B, H, N, D).to(q.dtype)

# =========================
# Main
# =========================
if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda"

    B, H, N, D = BATCH, NUM_HEADS, SEQLEN, HEAD_DIM
    assert D == 32 and H == 16 and N == 1024 and B == 1, "初始延迟测试参数需满足规范"

    q = torch.randn(B, H, N, D, device=device, dtype=DTYPE, requires_grad=True)
    k = torch.randn(B, H, N, D, device=device, dtype=DTYPE, requires_grad=True)
    v = torch.randn(B, H, N, D, device=device, dtype=DTYPE, requires_grad=True)

    # Correctness check vs SDPA (forward only)
    with torch.no_grad():
        ref = sdpa_reference(q, k, v, causal=CAUSAL)
        our = flash_attention(q, k, v, causal=CAUSAL)
        max_abs = (our - ref).abs().max().item()
    print(f"[Correctness] max_abs_diff_vs_sdpa: {max_abs:.4e}")

    # Latency: forward
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    f_metrics = measure_latency(lambda: flash_attention(q, k, v, causal=CAUSAL), warmup=10, iters=100)
    peak_mem_fwd = torch.cuda.max_memory_allocated()
    tokens = B * H * N
    print(f"[Latency|Forward] mean={f_metrics['mean_ms']:.3f} ms "
          f"(std={f_metrics['std_ms']:.3f}, iters={f_metrics['iters']}), "
          f"throughput={tokens / (f_metrics['mean_ms']/1e3):.1f} tokens/s, "
          f"peak_mem={peak_mem_fwd/1e6:.1f} MB")

    # Latency: end-to-end (fwd+bwd)
    def fwd_bwd():
        o = flash_attention(q, k, v, causal=CAUSAL)
        loss = o.float().pow(2).mean()
        loss.backward()
        for t in (q, k, v):
            if t.grad is not None:
                t.grad.zero_()
        return o

    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    e2e_metrics = measure_latency(fwd_bwd, warmup=10, iters=50)
    peak_mem_e2e = torch.cuda.max_memory_allocated()
    print(f"[Latency|Fwd+Bwd] mean={e2e_metrics['mean_ms']:.3f} ms "
          f"(std={e2e_metrics['std_ms']:.3f}, iters={e2e_metrics['iters']}), "
          f"throughput={tokens / (e2e_metrics['mean_ms']/1e3):.1f} tokens/s, "
          f"peak_mem={peak_mem_e2e/1e6:.1f} MB")

    # Max batch size search (optional)
    max_b = try_max_batch(base_B=1, H=H, N=N, D=D, dtype=DTYPE, causal=CAUSAL)
    print(f"[MaxBatch] max_batch_size at (H={H}, N={N}, D={D}, dtype={DTYPE}): {max_b}")

    # print("\n[Nsight Compute] 已添加 NVTX 标记：FA2_FWD / FA2_BWD。")
    # print("  使用示例：\n"
    #       "    ncu --set full --target-processes all "
    #       "-k _fwd_kernel,_bwd_kernel python FA2_triton_bench.py\n"
    #       "  在报告中查看 DRAM Bytes（dram__bytes.* 指标）以评估 Bandwidth。")
