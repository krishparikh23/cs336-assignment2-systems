import torch
import triton
import triton.testing
import pandas as pd
import math
import itertools
import gc
import warnings
from torch import Tensor
from torch.nn.functional import softmax
from einops import einsum

from cs336_systems.attention import FlashAttentionTriton

warnings.filterwarnings("ignore", category=Warning)

torch._functorch.config.donated_buffer=False

def scaled_dot_product_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """Scaled dot-product attention (PyTorch baseline)."""
    d_k = K.shape[-1]
    attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)

    if mask is not None:
        attention_scores = torch.where(mask, attention_scores, torch.tensor(float("-inf"), dtype=attention_scores.dtype, device=attention_scores.device))

    attention_weights = softmax(attention_scores, dim=-1)

    return einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")

def run_pytorch_fwd_bwd(q, k, v, mask, grad_o):
    o = scaled_dot_product_attention(q, k, v, mask)
    o.backward(grad_o, retain_graph=False)

def run_triton_fwd_bwd(q, k, v, grad_o, qts, kts):
    if q.grad is not None: q.grad.zero_()
    if k.grad is not None: k.grad.zero_()
    if v.grad is not None: v.grad.zero_()
    o = FlashAttentionTriton.apply(q, k, v, True, qts, kts)
    o.backward(grad_o, retain_graph=False)

def benchmark():
    seqlens = [2**i for i in range(7, 17)]
    d_models = [2**i for i in range(4, 8)]
    dtypes = [torch.bfloat16, torch.float32]
    device = torch.device("cuda")
    Q_TILE_SIZE = 32
    K_TILE_SIZE = 32

    print(f"Benchmarking...")
    print(f"SeqLens: {seqlens}")
    print(f"D_Models: {d_models}")
    print(f"Dtypes: {dtypes}")
    print("-" * 80)

    results = []

    for N, D, dtype in itertools.product(seqlens, d_models, dtypes):
        print(f"Config: SeqLen={N}, Dim={D}, dtype={dtype}")
        config_result = {'seq_len': N, 'd_model': D, 'dtype': str(dtype).split('.')[-1]}

        Q = torch.randn(1, N, D, dtype=dtype, device=device, requires_grad=True)
        K = torch.randn(1, N, D, dtype=dtype, device=device, requires_grad=True)
        V = torch.randn(1, N, D, dtype=dtype, device=device, requires_grad=True)
        grad_O = torch.randn(1, N, D, dtype=dtype, device=device)

        pt_causal_mask = torch.tril(torch.ones((N, N), device=device, dtype=torch.bool)).unsqueeze(0)

        torch.cuda.synchronize()

        print("Benchmarking PyTorch...")
        try:
            q_pt_fwd, k_pt_fwd, v_pt_fwd = Q.clone(), K.clone(), V.clone()
            q_pt_bwd, k_pt_bwd, v_pt_bwd, grad_o_pt_bwd = Q.clone(), K.clone(), V.clone(), grad_O.clone()
            q_pt_e2e, k_pt_e2e, v_pt_e2e, grad_o_pt_e2e = Q.clone(), K.clone(), V.clone(), grad_O.clone()

            mask_pt_fwd = pt_causal_mask.clone()
            mask_pt_bwd = pt_causal_mask.clone()
            mask_pt_e2e = pt_causal_mask.clone()

            ms_fwd_pt = triton.testing.do_bench(lambda: scaled_dot_product_attention(q_pt_fwd, k_pt_fwd, v_pt_fwd, mask_pt_fwd), quantiles=[0.5])
            torch.cuda.synchronize()
            del q_pt_fwd, k_pt_fwd, v_pt_fwd, mask_pt_fwd

            o_pt_for_bwd = scaled_dot_product_attention(q_pt_bwd, k_pt_bwd, v_pt_bwd, mask_pt_bwd)
            torch.cuda.synchronize()

            ms_bwd_pt = triton.testing.do_bench(lambda: o_pt_for_bwd.backward(grad_o_pt_bwd, retain_graph=True), quantiles=[0.5])
            o_pt_for_bwd.backward(grad_o_pt_bwd, retain_graph=False)
            torch.cuda.synchronize()
            del o_pt_for_bwd, q_pt_bwd, k_pt_bwd, v_pt_bwd, grad_o_pt_bwd, mask_pt_bwd

            ms_e2e_pt = triton.testing.do_bench(lambda: run_pytorch_fwd_bwd(q_pt_e2e, k_pt_e2e, v_pt_e2e, mask_pt_e2e, grad_o_pt_e2e), quantiles=[0.5])
            torch.cuda.synchronize()
            del q_pt_e2e, k_pt_e2e, v_pt_e2e, grad_o_pt_e2e, mask_pt_e2e

            config_result['pytorch_fwd_ms'] = ms_fwd_pt
            config_result['pytorch_bwd_ms'] = ms_bwd_pt
            config_result['pytorch_e2e_ms'] = ms_e2e_pt
            config_result['pytorch_status'] = 'OK'
            print(f"PyTorch OK: Fwd={ms_fwd_pt:.3f}ms, Bwd={ms_bwd_pt:.3f}ms, E2E={ms_e2e_pt:.3f}ms")

        except torch.cuda.OutOfMemoryError:
            print("PyTorch OOM")
            config_result['pytorch_fwd_ms'] = float('nan')
            config_result['pytorch_bwd_ms'] = float('nan')
            config_result['pytorch_e2e_ms'] = float('nan')
            config_result['pytorch_status'] = 'OOM'

        print("Benchmarking Triton...")
        try:
            q_tr_fwd, k_tr_fwd, v_tr_fwd = Q.clone(), K.clone(), V.clone()
            q_tr_bwd, k_tr_bwd, v_tr_bwd, grad_o_tr_bwd = Q.clone(), K.clone(), V.clone(), grad_O.clone()
            q_tr_e2e, k_tr_e2e, v_tr_e2e, grad_o_tr_e2e = Q.clone(), K.clone(), V.clone(), grad_O.clone()

            ms_fwd_tr = triton.testing.do_bench(lambda: FlashAttentionTriton.apply(q_tr_fwd, k_tr_fwd, v_tr_fwd, True, Q_TILE_SIZE, K_TILE_SIZE), quantiles=[0.5])
            torch.cuda.synchronize()
            del q_tr_fwd, k_tr_fwd, v_tr_fwd

            o_tr_for_bwd = FlashAttentionTriton.apply(q_tr_bwd, k_tr_bwd, v_tr_bwd, True, Q_TILE_SIZE, K_TILE_SIZE)
            torch.cuda.synchronize()

            ms_bwd_tr = triton.testing.do_bench(lambda: o_tr_for_bwd.backward(grad_o_tr_bwd, retain_graph=True), quantiles=[0.5])
            o_tr_for_bwd.backward(grad_o_tr_bwd, retain_graph=False)
            torch.cuda.synchronize()
            del o_tr_for_bwd,q_tr_bwd, k_tr_bwd, v_tr_bwd, grad_o_tr_bwd

            ms_e2e_tr = triton.testing.do_bench(lambda: run_triton_fwd_bwd(q_tr_e2e, k_tr_e2e, v_tr_e2e, grad_o_tr_e2e, Q_TILE_SIZE, K_TILE_SIZE), quantiles=[0.5])
            torch.cuda.synchronize()
            del q_tr_e2e, k_tr_e2e, v_tr_e2e, grad_o_tr_e2e

            config_result['triton_fwd_ms'] = ms_fwd_tr
            config_result['triton_bwd_ms'] = ms_bwd_tr
            config_result['triton_e2e_ms'] = ms_e2e_tr
            config_result['triton_status'] = 'OK'
            print(f"Triton OK: Fwd={ms_fwd_tr:.3f}ms, Bwd={ms_bwd_tr:.3f}ms, E2E={ms_e2e_tr:.3f}ms")

        except torch.cuda.OutOfMemoryError:
            print("Triton OOM")
            config_result['triton_fwd_ms'] = float('nan')
            config_result['triton_bwd_ms'] = float('nan')
            config_result['triton_e2e_ms'] = float('nan')
            config_result['triton_status'] = 'OOM'

        results.append(config_result)
        del Q, K, V, grad_O, pt_causal_mask
        gc.collect()
        torch.cuda.empty_cache()
        print("-" * 80)

    results_df = pd.DataFrame(results)

    column_order = [
        'seq_len', 'd_model', 'dtype',
        'pytorch_fwd_ms', 'pytorch_bwd_ms', 'pytorch_e2e_ms', 'pytorch_status',
        'triton_fwd_ms', 'triton_bwd_ms', 'triton_e2e_ms', 'triton_status'
    ]
    results_df = results_df[column_order]

    print("\n=== Benchmark Results ===")
    print(results_df.to_markdown(index=False, floatfmt=".3f"))

if __name__ == "__main__":
    benchmark() 