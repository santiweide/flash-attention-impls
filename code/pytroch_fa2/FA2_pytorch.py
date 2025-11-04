import argparse
import math
import time
import torch
import torch.nn.functional as F

def try_import_fa2():
    try:
        from flash_attn.flash_attn_interface import flash_attn_qkvpacked_func
        return flash_attn_qkvpacked_func
    except Exception:
        return None

def run_once_fa2(qkv, causal=False, dropout_p=0.0, softmax_scale=None):
    # qkv: [B, S, 3, H, D]
    flash_attn_qkvpacked_func = try_import_fa2()
    if flash_attn_qkvpacked_func is None:
        raise RuntimeError("flash-attn not available")
    return flash_attn_qkvpacked_func(
        qkv, dropout_p=dropout_p, softmax_scale=softmax_scale, causal=causal
    )

def run_once_sdpa(q, k, v, causal=False, scale=None):
    # q,k,v: [B, H, S, D]
    # PyTorch SDPA requires shape [B, H, S, D]
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=scale)

def bench_once(
    batch_size=1, num_heads=32, seqlen=8192, head_dim=128,
    causal=False, use_bfloat16=True, iters=100, warmup=10, use_fa2=True
):
    assert torch.cuda.is_available(), "CUDA is required"
    device = "cuda"

    dtype = torch.bfloat16 if use_bfloat16 else torch.float16
    # Generate input tensors
    # For FA2: use qkv packed shape [B, S, 3, H, D]
    # For SDPA: use [B, H, S, D]
    torch.manual_seed(0)
    q = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype)
    qkv = torch.stack([q, k, v], dim=2)  # [B, S, 3, H, D]

    # Choose backend
    fa2_callable = try_import_fa2()
    path = "FA2" if (use_fa2 and fa2_callable is not None) else "SDPA"
    # SDPA requires [B, H, S, D]
    q_t = q.permute(0, 2, 1, 3).contiguous()
    k_t = k.permute(0, 2, 1, 3).contiguous()
    v_t = v.permute(0, 2, 1, 3).contiguous()

    # Warmup
    torch.cuda.synchronize()
    with torch.cuda.amp.autocast(enabled=True, dtype=dtype):
        for _ in range(warmup):
            if path == "FA2":
                out = run_once_fa2(qkv, causal=causal)
            else:
                out = run_once_sdpa(q_t, k_t, v_t, causal=causal)
    torch.cuda.synchronize()

    # Timed run
    t0 = time.perf_counter()
    with torch.cuda.amp.autocast(enabled=True, dtype=dtype):
        for _ in range(iters):
            if path == "FA2":
                out = run_once_fa2(qkv, causal=causal)
            else:
                out = run_once_sdpa(q_t, k_t, v_t, causal=causal)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    # Normalize output tensor (prevent optimization removal)
    total_sum = float(out.float().sum().item())

    avg_ms = (t1 - t0) * 1000.0 / iters
    # Approximate throughput (token/s): B*S / (seconds/iter)
    tokens_per_iter = batch_size * seqlen
    throughput = tokens_per_iter / (avg_ms / 1000.0)

    # Memory usage
    max_mem = torch.cuda.max_memory_allocated(device) / (1024**2)

    return {
        "path": path,
        "avg_ms": avg_ms,
        "throughput_tokens_per_s": throughput,
        "total_sum_guard": total_sum,
        "max_memory_MB": max_mem,
    }

def find_max_batch(
    start_bsz, num_heads, seqlen, head_dim, use_fa2=True, use_bfloat16=True, causal=False
):
    bsz = start_bsz
    last_ok = None
    while True:
        try:
            torch.cuda.reset_peak_memory_stats()
            _ = bench_once(
                batch_size=bsz, num_heads=num_heads, seqlen=seqlen, head_dim=head_dim,
                use_fa2=use_fa2, use_bfloat16=use_bfloat16, causal=causal, iters=5, warmup=2
            )
            last_ok = bsz
            bsz *= 2
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "cuda error" in str(e).lower():
                return last_ok
            else:
                raise

def pretty_print(res, b, h, s, d):
    print("="*72)
    print(f"Path              : {res['path']}")
    print(f"Shape             : B={b}, H={h}, S={s}, D={d} (model_dim = {h*d})")
    print(f"Avg latency       : {res['avg_ms']:.3f} ms / iter")
    print(f"Throughput        : {res['throughput_tokens_per_s']:.2f} tokens/s")
    print(f"Max CUDA memory   : {res['max_memory_MB']:.1f} MB")
    print(f"Guard value (sum) : {res['total_sum_guard']:.3f}")
    print("="*72)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--seqlen", type=int, default=8192)
    parser.add_argument("--head-dim", type=int, default=128,
                        help="Per-head dimension (your hidden_dim=128 is interpreted as head_dim here)")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 (default: BF16)")
    parser.add_argument("--causal", action="store_true", help="Enable causal mask")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--no-fa2", action="store_true", help="Force SDPA instead of FA2")
    parser.add_argument("--find-max-batch", action="store_true", help="Automatically search for the maximum stable batch size")
    args = parser.parse_args()

    use_bfloat16 = not args.fp16
    use_fa2 = not args.no_fa2

    if not torch.cuda.is_available():
        raise SystemError("A CUDA-capable GPU is required to run this script.")

    # Run one benchmark
    torch.cuda.reset_peak_memory_stats()
    res = bench_once(
        batch_size=args.batch_size,
        num_heads=args.num_heads,
        seqlen=args.seqlen,
        head_dim=args.head_dim,
        causal=args.causal,
        use_bfloat16=use_bfloat16,
        iters=args.iters,
        warmup=args.warmup,
        use_fa2=use_fa2,
    )
    pretty_print(res, args.batch_size, args.num_heads, args.seqlen, args.head_dim)

    # Search for max batch size
    if args.find_max_batch:
        try:
            max_b = find_max_batch(
                max(1, args.batch_size), args.num_heads, args.seqlen, args.head_dim,
                use_fa2=use_fa2, use_bfloat16=use_bfloat16, causal=args.causal
            )
            print(f"[Max Batch Size] For the same shape (H={args.num_heads}, S={args.seqlen}, D={args.head_dim}), "
                  f"the maximum stable forward batch size ≈ {max_b}")
        except RuntimeError as e:
            print(f"[Max Batch Size] Exception occurred during search: {e}")

if __name__ == "__main__":
    main()
