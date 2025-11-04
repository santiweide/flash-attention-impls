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
    times = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    with torch.cuda.amp.autocast(enabled=True, dtype=dtype):
        for _ in range(iters):
            start_event.record()
            if path == "FA2":
                out = run_once_fa2(qkv, causal=causal)
            else:
                out = run_once_sdpa(q_t, k_t, v_t, causal=causal)
            end_event.record()
            torch.cuda.synchronize()
            times.append(start_event.elapsed_time(end_event))

    # Normalize output tensor (prevent optimization removal)
    total_sum = float(out.float().sum().item())

    import statistics as stats
    avg_ms = stats.mean(times)
    std_ms = stats.pstdev(times)
    
    # Approximate throughput (token/s): B*S / (seconds/iter)
    tokens_per_iter = batch_size * seqlen
    throughput = tokens_per_iter / (avg_ms / 1000.0)

    # Memory usage
    max_mem = torch.cuda.max_memory_allocated(device) / (1024**2)

    return {
        "path": path,
        "avg_ms": avg_ms,
        "std_ms": std_ms,
        "throughput_tokens_per_s": throughput,
        "total_sum_guard": total_sum,
        "max_memory_MB": max_mem,
    }

def calculate_flops(B, H, N, D):
    """
    Calculate FLOPs for attention (forward pass approximation)
    """
    # Q@K^T: B*H*N*N*D
    # Softmax: negligible 
    # P@V: B*H*N*N*D
    # Total ≈ 2*B*H*N^2*D (simplified)
    return 4 * B * H * N * N * D

def find_max_batch(
    start_bsz, num_heads, seqlen, head_dim, use_fa2=True, use_bfloat16=True, causal=False
):
    bsz = start_bsz
    last_ok = None
    while True:
        try:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
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

def main():
    if not torch.cuda.is_available():
        raise SystemError("A CUDA-capable GPU is required to run this script.")

    # Test configurations (B, H, N, D)
    test_configs = [
        (1, 1, 512, 64),
        (1, 1, 1024, 64),
        (1, 1, 2048, 64),
        (1, 1, 4096, 64),
        (1, 1, 8192, 64),
        (1, 32, 8192, 32),
        (1, 32, 8192, 64),
        (1, 32, 8192, 128),
    ]

    # Test settings
    use_bfloat16 = False  # Use FP16 for consistency
    causal = True
    iters = 100
    warmup = 10

    print("="*100)
    print("Flash Attention / SDPA Performance Benchmark")
    print("="*100)
    print(f"Configuration: dtype={'bf16' if use_bfloat16 else 'fp16'}, causal={causal}")
    print(f"Iterations: warmup={warmup}, iters={iters}")
    print("="*100)
    print()

    # Test both FA2 and SDPA
    backends = [
        ("FA2", True),
        ("SDPA", False),
    ]

    all_results = {}

    for backend_name, use_fa2 in backends:
        print(f"\n{'='*100}")
        print(f"Testing Backend: {backend_name}")
        print(f"{'='*100}\n")
        
        results = []
        
        for config_idx, (B, H, N, D) in enumerate(test_configs, 1):
            print(f"[{config_idx}/{len(test_configs)}] Testing Config (B={B}, H={H}, N={N}, D={D})")
            print("-" * 80)

            try:
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
                
                res = bench_once(
                    batch_size=B,
                    num_heads=H,
                    seqlen=N,
                    head_dim=D,
                    causal=causal,
                    use_bfloat16=use_bfloat16,
                    iters=iters,
                    warmup=warmup,
                    use_fa2=use_fa2,
                )
                
                # Calculate TFLOPs/s
                flops = calculate_flops(B, H, N, D)
                time_s = res['avg_ms'] / 1000.0
                tflops = (flops / time_s) / 1e12
                
                print(f"  Latency:    {res['avg_ms']:.2f} ms ± {res['std_ms']:.2f}")
                print(f"  Throughput: {tflops:.2f} TFLOPs/s")
                print(f"  Tokens/s:   {res['throughput_tokens_per_s']:.2f}")
                print(f"  Memory:     {res['max_memory_MB']:.1f} MB")
                
                results.append({
                    'config': f"({B}, {H}, {N}, {D})",
                    'B': B, 'H': H, 'N': N, 'D': D,
                    'avg_ms': res['avg_ms'],
                    'std_ms': res['std_ms'],
                    'tflops': tflops,
                    'tokens_per_s': res['throughput_tokens_per_s'],
                    'memory_mb': res['max_memory_MB'],
                })
                
            except RuntimeError as e:
                print(f"  FAILED: {e}")
                results.append({
                    'config': f"({B}, {H}, {N}, {D})",
                    'B': B, 'H': H, 'N': N, 'D': D,
                    'avg_ms': float('inf'),
                    'std_ms': 0,
                    'tflops': 0,
                    'tokens_per_s': 0,
                    'memory_mb': 0,
                })
            
            print()
        
        all_results[backend_name] = results

    # Print summary comparison table
    print("\n" + "="*100)
    print("SUMMARY COMPARISON TABLE")
    print("="*100)
    print(f"{'Config (B,H,N,D)':<20} {'Backend':<8} {'Latency(ms)':<14} {'TFLOPs/s':<12} {'Tokens/s':<15} {'Memory(MB)':<12}")
    print("-"*100)
    
    for backend_name in ["FA2", "SDPA"]:
        if backend_name in all_results:
            for r in all_results[backend_name]:
                if r['avg_ms'] != float('inf'):
                    print(f"{r['config']:<20} {backend_name:<8} "
                          f"{r['avg_ms']:>10.2f}     "
                          f"{r['tflops']:>10.2f}   "
                          f"{r['tokens_per_s']:>13.2f}   "
                          f"{r['memory_mb']:>10.1f}")
                else:
                    print(f"{r['config']:<20} {backend_name:<8} {'FAILED':<14}")
    
    print("="*100)
    
    # Speedup comparison
    if "FA2" in all_results and "SDPA" in all_results:
        print("\n" + "="*100)
        print("SPEEDUP COMPARISON (FA2 vs SDPA)")
        print("="*100)
        print(f"{'Config (B,H,N,D)':<20} {'FA2 Latency':<15} {'SDPA Latency':<15} {'Speedup':<10}")
        print("-"*100)
        
        for i, config in enumerate(test_configs):
            fa2_ms = all_results["FA2"][i]['avg_ms']
            sdpa_ms = all_results["SDPA"][i]['avg_ms']
            
            if fa2_ms != float('inf') and sdpa_ms != float('inf') and fa2_ms > 0:
                speedup = sdpa_ms / fa2_ms
                config_str = f"({config[0]}, {config[1]}, {config[2]}, {config[3]})"
                print(f"{config_str:<20} {fa2_ms:>13.2f}   {sdpa_ms:>13.2f}   {speedup:>8.2f}x")
            else:
                config_str = f"({config[0]}, {config[1]}, {config[2]}, {config[3]})"
                print(f"{config_str:<20} {'N/A':<15} {'N/A':<15} {'N/A':<10}")
        
        print("="*100)

    # Optional: Find max batch size for one representative config
    print("\n" + "="*100)
    print("MAX BATCH SIZE SEARCH (for config: B=1, H=32, N=8192, D=128)")
    print("="*100)
    
    for backend_name, use_fa2 in backends:
        try:
            max_b = find_max_batch(
                start_bsz=1, num_heads=32, seqlen=8192, head_dim=128,
                use_fa2=use_fa2, use_bfloat16=use_bfloat16, causal=causal
            )
            print(f"{backend_name}: Maximum stable batch size ≈ {max_b}")
        except Exception as e:
            print(f"{backend_name}: Search failed - {e}")
    
    print("="*100)

if __name__ == "__main__":
    main()