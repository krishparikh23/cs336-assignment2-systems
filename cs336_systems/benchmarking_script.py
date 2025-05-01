import argparse
import timeit
import torch
import torch.nn as nn
import numpy as np
import sys
import os
import torch.cuda.nvtx as nvtx
import math
from torch import Tensor
from jaxtyping import Float, Bool
from einops import einsum
from contextlib import nullcontext

from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy, softmax

VOCAB_SIZE = 10000
BATCH_SIZE = 4
ROPE_THETA = 10000.0

MODEL_CONFIGS = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
    "2.7B": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}

def annotated_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys    d_k"],
    V: Float[Tensor, " ... keys    d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    with nvtx.range("scaled dot product attention"):
        d_k = K.shape[-1]
        with nvtx.range("computing attention scores"):
            attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)

        with nvtx.range("computing softmax"):
            if mask is not None:
                attention_scores = torch.where(mask, attention_scores, torch.tensor(float("-inf"), device=Q.device, dtype=Q.dtype))
            attention_weights = softmax(attention_scores, dim=-1)

        with nvtx.range("final matmul"):
            result = einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")
        return result


def benchmark(
    model_size: str,
    context_length: int,
    warmup_steps: int,
    num_steps: int,
    mode: str,
    device: str,
    mixed_precision: bool,
):
    print(f"Using device: {device}")

    import cs336_basics.model
    from cs336_basics.model import BasicsTransformerLM
    cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention

    config = MODEL_CONFIGS[model_size]
    print(f"Initializing model: {model_size} (d_model={config['d_model']}, n_layers={config['num_layers']}, n_heads={config['num_heads']}, context={context_length})")

    model = BasicsTransformerLM(
        vocab_size=VOCAB_SIZE,
        context_length=context_length,
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        rope_theta=ROPE_THETA,
    ).to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=1e-4)

    print(f"Generating random data: batch_size={BATCH_SIZE}, context_length={context_length}")
    inputs = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, context_length), device=device, dtype=torch.long)
    targets = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, context_length), device=device, dtype=torch.long)

    amp_context = nullcontext()
    if mixed_precision and device == 'cuda':
        print("Using mixed precision (BF16)")
        amp_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    elif mixed_precision and device == 'cpu':
        print("Warning: Mixed precision (BF16) requested on CPU, using FP32 instead.")

    print(f"Running {warmup_steps} warm-up steps...")
    with nvtx.range("Warm-up"):
        for _ in range(warmup_steps):
            if mode == 'forward':
                with amp_context:
                    _ = model(inputs)
            elif mode == 'forward_backward':
                optimizer.zero_grad()
                with amp_context:
                    outputs = model(inputs)
                loss = cross_entropy(outputs.view(-1, VOCAB_SIZE), targets.view(-1))
                loss.backward()
            elif mode == 'full_step':
                optimizer.zero_grad()
                with amp_context:
                    outputs = model(inputs)
                loss = cross_entropy(outputs.view(-1, VOCAB_SIZE), targets.view(-1))
                loss.backward()
                optimizer.step()
            else:
                 raise ValueError(f"Invalid mode: {mode}")

            if device == 'cuda':
                torch.cuda.synchronize()

    print(f"Running {num_steps} measurement steps (mode: {mode})...")
    forward_times = []
    backward_times = []
    step_times = []

    with nvtx.range("Measurement"):
        for _ in range(num_steps):
            if device == 'cuda':
                torch.cuda.synchronize()

            if mode == 'forward':
                with nvtx.range("Forward Pass"):
                    start_forward = timeit.default_timer()
                    with amp_context:
                        _ = model(inputs)
                    if device == 'cuda':
                        torch.cuda.synchronize()
                    end_forward = timeit.default_timer()
                forward_times.append(end_forward - start_forward)

            elif mode == 'forward_backward':
                optimizer.zero_grad()

                with nvtx.range("Forward Pass"):
                    start_forward = timeit.default_timer()
                    with amp_context:
                        outputs = model(inputs)
                    if device == 'cuda':
                         torch.cuda.synchronize()
                    end_forward = timeit.default_timer()
                forward_times.append(end_forward - start_forward)

                loss = cross_entropy(outputs.view(-1, VOCAB_SIZE), targets.view(-1))

                with nvtx.range("Backward Pass"):
                    start_backward = timeit.default_timer()
                    loss.backward()
                    if device == 'cuda':
                        torch.cuda.synchronize()
                    end_backward = timeit.default_timer()
                backward_times.append(end_backward - start_backward)

            elif mode == 'full_step':
                optimizer.zero_grad()

                with nvtx.range("Forward Pass"):
                    start_forward = timeit.default_timer()
                    with amp_context:
                        outputs = model(inputs)
                    if device == 'cuda':
                         torch.cuda.synchronize()
                    end_forward = timeit.default_timer()
                forward_times.append(end_forward - start_forward)

                loss = cross_entropy(outputs.view(-1, VOCAB_SIZE), targets.view(-1))

                with nvtx.range("Backward Pass"):
                    start_backward = timeit.default_timer()
                    loss.backward()
                    if device == 'cuda':
                        torch.cuda.synchronize()
                    end_backward = timeit.default_timer()
                backward_times.append(end_backward - start_backward)

                with nvtx.range("Optimizer Step"):
                    start_step = timeit.default_timer()
                    optimizer.step()
                    if device == 'cuda':
                        torch.cuda.synchronize()
                    end_step = timeit.default_timer()
                step_times.append(end_step - start_step)

    avg_forward, std_forward = 0.0, 0.0
    avg_backward, std_backward = 0.0, 0.0
    avg_step, std_step = 0.0, 0.0
    avg_total = 0.0

    if forward_times:
        avg_forward = np.mean(forward_times)
        std_forward = np.std(forward_times)
        avg_total += avg_forward
    if backward_times:
        avg_backward = np.mean(backward_times)
        std_backward = np.std(backward_times)
        avg_total += avg_backward
    if step_times:
        avg_step = np.mean(step_times)
        std_step = np.std(step_times)
        avg_total += avg_step

    print("=== Benchmark Results ===")
    print(f"Mode:               {mode}")
    print(f"Model Size:         {model_size}")
    print(f"Context Length:     {context_length}")
    print(f"Warm-up Steps:      {warmup_steps}")
    print(f"Number of Steps:    {num_steps}")

    if forward_times:
        print(f"Avg Forward Time:   {avg_forward:.6f} sec (std: {std_forward:.6f})")
    if backward_times:
        print(f"Avg Backward Time:  {avg_backward:.6f} sec (std: {std_backward:.6f})")
    if step_times:
        print(f"Avg Step Time:      {avg_step:.6f} sec (std: {std_step:.6f})")
    print(f"Avg Total Time:     {avg_total:.6f} sec")
    tokens_per_step = BATCH_SIZE * context_length
    print(f"Throughput:         {tokens_per_step / avg_total:.2f} tokens/sec")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", type=str, required=True, choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--context_length", type=int, required=True)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--num_steps", type=int, default=10)
    parser.add_argument("--mode", type=str, required=True, choices=["forward", "forward_backward", "full_step"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", choices=["cuda", "cpu"])
    parser.add_argument("--mixed_precision", action='store_true')

    args = parser.parse_args()
    benchmark(**vars(args))