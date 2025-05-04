import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import os
from functools import partial
import argparse

from cs336_systems.sharded_optimizer import ShardedOptimizer
from cs336_systems.ddp import DDPBucketed as DDP
from cs336_basics.model import BasicsTransformerLM
from torch.optim import AdamW

xl_config = {
    "d_model": 1600,
    "d_ff": 6400,
    "num_layers": 48,
    "num_heads": 25,
    "vocab_size": 10000,
    "context_length": 512,
    "rope_theta": 10000.0,
}

bucket_size_mb = 100
learning_rate = 1e-4
num_warmup_steps = 5
num_measured_steps = 5

def setup(rank):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=2)

def cleanup():
    dist.destroy_process_group()

def worker(rank, optimizer_type, results_dict):
    setup(rank)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    torch.cuda.reset_peak_memory_stats(device)

    model = BasicsTransformerLM(**xl_config).to(device)
    model = DDP(model, bucket_size_mb=100)

    torch.cuda.synchronize()
    model_init_peak_mem = torch.cuda.max_memory_allocated(device)

    input_ids = torch.randint(0, xl_config["vocab_size"], (1, xl_config["context_length"]), device=device)

    def train_step(model, optimizer, input_ids):
        optimizer.zero_grad()
        outputs = model(input_ids)
        loss = outputs.sum()
        loss.backward()
        model.finish_gradient_synchronization()
        torch.cuda.synchronize()
        before_step_peak_mem = torch.cuda.max_memory_allocated(device)
        optimizer.step()
        torch.cuda.synchronize()
        after_step_peak_mem = torch.cuda.max_memory_allocated(device)
        return loss, before_step_peak_mem, after_step_peak_mem

    def profile_optimizer(optimizer_class, model_module):
        """ Profiles a given optimizer class. """
        if optimizer_class == ShardedOptimizer:
             params_to_optimize = model_module.parameters()
             optimizer = optimizer_class(params_to_optimize, optimizer_cls=AdamW, lr=learning_rate)
        else:
             params_to_optimize = model.parameters()
             optimizer = optimizer_class(params_to_optimize, lr=learning_rate)

        torch.cuda.reset_peak_memory_stats(device)
        for step in range(num_warmup_steps):
            print(f"Rank {rank}: Warmup step {step+1}/{num_warmup_steps}")
            train_step(model, optimizer, input_ids)
        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats(device)
        start_time = time.time()
        max_before_step_peak_mem = 0
        max_after_step_peak_mem = 0
        print(f"Rank {rank}: Starting measurement...")
        for i in range(num_measured_steps):
            print(f"Rank {rank}: Measuring step {i+1}/{num_measured_steps}")
            _, before_step_peak, after_step_peak = train_step(model, optimizer, input_ids)
            max_before_step_peak_mem = max(max_before_step_peak_mem, before_step_peak)
            max_after_step_peak_mem = max(max_after_step_peak_mem, after_step_peak)

        torch.cuda.synchronize()
        end_time = time.time()
        print(f"Rank {rank}: Measurement finished.")

        measurement_phase_peak_mem = torch.cuda.max_memory_allocated(device)
        avg_time_per_iter = (end_time - start_time) / num_measured_steps

        del optimizer
        torch.cuda.empty_cache()

        return measurement_phase_peak_mem, avg_time_per_iter, max_before_step_peak_mem, max_after_step_peak_mem

    overall_peak_mem = 0
    avg_time = 0
    before_step_peak = 0
    after_step_peak = 0

    if optimizer_type == 'standard':
        print(f"Rank {rank}: Profiling Standard AdamW...")
        overall_peak_mem, avg_time, before_step_peak, after_step_peak = profile_optimizer(
            AdamW, model
        )
        print(f"Rank {rank}: Finished Standard AdamW profiling.")
    elif optimizer_type == 'sharded':
        print(f"Rank {rank}: Profiling Sharded Optimizer...")
        overall_peak_mem, avg_time, before_step_peak, after_step_peak = profile_optimizer(
             ShardedOptimizer, model
        )
        print(f"Rank {rank}: Finished Sharded Optimizer profiling.")

    results = {
        "model_init_peak_mem": model_init_peak_mem,
        "overall_peak_mem": overall_peak_mem,
        "before_step_peak": before_step_peak,
        "after_step_peak": after_step_peak,
        "avg_time": avg_time,
        "optimizer_type": optimizer_type
    }

    gathered_results = [None] * 2

    dist.gather_object(results, gathered_results if rank == 0 else None, dst=0)

    if rank == 0:
         valid_results = [r for r in gathered_results]
         if valid_results:
             results_dict['gathered'] = valid_results

    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--optimizer_type',
        type=str,
        required=True,
        choices=['standard', 'sharded'],
    )
    args = parser.parse_args()

    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    results_dict = manager.dict()

    print(f"Starting benchmark for '{args.optimizer_type}'")

    spawn_context = mp.spawn(
        partial(worker, optimizer_type=args.optimizer_type, results_dict=results_dict),
        nprocs=2,
        join=True
    )

    gathered_results = results_dict['gathered']
    print("=== Profiling Results ===")

    total_model_init_mem_gb = sum(r['model_init_peak_mem'] for r in gathered_results) / (1024**3)
    total_overall_peak_gb = sum(r['overall_peak_mem'] for r in gathered_results) / (1024**3)
    total_before_step_peak_gb = sum(r['before_step_peak'] for r in gathered_results) / (1024**3)
    total_after_step_peak_gb = sum(r['after_step_peak'] for r in gathered_results) / (1024**3)
    avg_time = sum(r['avg_time'] for r in gathered_results) / 2

    import pandas as pd

    results_df = pd.DataFrame({
        'Model Init': [total_model_init_mem_gb],
        'Before opt.step': [total_before_step_peak_gb],
        'After opt.step': [total_after_step_peak_gb],
        'Overall Peak': [total_overall_peak_gb],
        'Avg Time/Iter (s)': [avg_time]
    })
    print(results_df.to_markdown(index=False, floatfmt=".4f"))