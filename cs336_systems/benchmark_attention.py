import torch
import torch.nn.functional as F
import math
import time
import itertools
import pandas as pd
import sys
from torch import compile as torch_compile

from torch import Tensor
from einops import einsum
from torch.nn.functional import softmax

# @torch_compile
def scaled_dot_product_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """Scaled dot-product attention.

    This function implements Eq. 1 of the Transformer paper.

    Args:
        Q: Tensor of queries, may have any number of leading dimensions.
        K: Tensor of keys, sharing leading dimensions with Q.
        V: Tensor of values, sharding leading dimensions with Q and K.
        mask: An (optional) mask of shape (..., seq_len, seq_len).
            Attention scores for positions with a mask value of `False` should
            be masked out, i.e., not affect the softmaxed attention probabilities.

    Returns:
        torch.FloatTensor of shape (..., seq_len, value_dimension)
        with the output of running your scaled dot product attention
        implementation with the provided key, query, and value tensors.
    """

    d_k = K.shape[-1]
    attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)

    if mask is not None:
        attention_scores = torch.where(mask, attention_scores, float("-inf"))

    attention_weights = softmax(attention_scores, dim=-1)

    return einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")

def benchmark_single(
    d_model: int,
    seq_len: int,
    batch_size: int,
    device: torch.device,
    warmup_iters: int,
    measure_iters: int,
) -> dict:
    """Runs the attention benchmark for a single configuration."""

    print(f"Benchmarking: d_model={d_model}, seq_len={seq_len}...")

    forward_time_ms = 0
    backward_time_ms = 0
    peak_memory_gb = 0
    status = "OK"

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    try:

        QKVs = [(torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True),
                    torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True),
                    torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True))
                for _ in range(measure_iters)]
        Os = []
        grad_O = torch.randn(batch_size, seq_len, d_model, device=device)
        
        for _ in range(warmup_iters):
            Q, K, V = QKVs[0]
            O = scaled_dot_product_attention(Q, K, V)
            O.backward(grad_O)

        torch.cuda.synchronize(device=device)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device=device)


        for Q, K, V in QKVs:
            start_event.record()
            output = scaled_dot_product_attention(Q, K, V)
            end_event.record()
            torch.cuda.synchronize(device=device)
            forward_time_ms += start_event.elapsed_time(end_event)
            Os.append(output)

        forward_time_ms /= measure_iters
        
        peak_memory_bytes = torch.cuda.max_memory_allocated(device=device)
        peak_memory_gb = peak_memory_bytes / (1024 ** 3)


        
        for O in Os:

            start_event.record()
            O.backward(grad_O, retain_graph=False)
            end_event.record()
            torch.cuda.synchronize(device=device)
            backward_time_ms += start_event.elapsed_time(end_event)

        backward_time_ms /= measure_iters

        del Q, K, V, O, QKVs, Os, grad_O
        torch.cuda.empty_cache()
        print(f"Done. Fwd: {forward_time_ms:.3f} ms, Bwd: {backward_time_ms:.3f} ms, Peak Mem: {peak_memory_gb:.3f} GB")

    except torch.cuda.OutOfMemoryError:
        status = "OOM"
        print(f"OOM for d_model={d_model}, seq_len={seq_len}")
        
        for var_name in ['Q', 'K', 'V', 'O', 'QKVs', 'Os', 'grad_O']:
             if var_name in locals():
                 try:
                     del locals()[var_name]
                 except NameError:
                     pass
        torch.cuda.empty_cache()

    
    return {
        "d_model": d_model,
        "seq_len": seq_len,
        "forward_ms": float('nan') if status == "OOM" else forward_time_ms,
        "backward_ms": float('nan') if status == "OOM" else backward_time_ms,
        "peak_memory_gb": float('nan') if status == "OOM" else peak_memory_gb,
        "status": status,
    }


if __name__ == "__main__":

    
    device = torch.device("cuda")
    batch_size = 8
    d_model_values = [16, 32, 64, 128]
    seq_len_values = [256, 1024, 4096, 8192, 16384]
    num_repeats = 100
    num_warmup = 5

    results = []

    print(f"Starting benchmark loop with batch_size={batch_size}, num_repeats={num_repeats}, num_warmup={num_warmup}...")
    print("-" * 80)

    

    for d_model, seq_len in itertools.product(d_model_values, seq_len_values):
        single_result = benchmark_single(
            d_model=d_model,
            seq_len=seq_len,
            batch_size=batch_size,
            device=device,
            warmup_iters=num_warmup,
            measure_iters=num_repeats
        )
        results.append(single_result)


    
    results_df = pd.DataFrame(results)
    
    results_df = results_df.round(3)
    results_df['peak_memory_gb'] = results_df['peak_memory_gb'].round(3)
    

    print("Benchmark Results:")
    
    print(results_df.to_markdown(index=False, floatfmt=".3f"))