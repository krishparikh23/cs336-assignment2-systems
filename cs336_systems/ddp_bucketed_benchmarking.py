import argparse
import time
import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import Adam
import torch.multiprocessing as mp

from cs336_basics.model import BasicsTransformerLM
from cs336_systems.ddp import DDPBucketed


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def main(rank, world_size, args):
    local_rank = rank
    setup(rank, world_size)
    device = torch.device(f"cuda:{local_rank}")

    xl_config = {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25}
    context_length = 512
    vocab_size = 10000
    batch_size = 1

    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        rope_theta=10000.0,
        **xl_config
    ).to(device)

    model = DDPBucketed(model, bucket_size_mb=args.bucket_size_mb).to(device)

    optimizer = Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    inputs = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    targets = torch.randint(0, vocab_size, (batch_size, context_length), device=device)

    num_steps = 10
    warmup_steps = 5
    step_times = []
    comm_times = []

    dist.barrier()
    torch.cuda.synchronize(device=device)

    step_start = torch.cuda.Event(enable_timing=True)
    step_end = torch.cuda.Event(enable_timing=True)
    comm_step_start = torch.cuda.Event(enable_timing=True)
    comm_step_end = torch.cuda.Event(enable_timing=True)

    for step in range(num_steps):
        step_start.record()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
        loss.backward()

        comm_step_start.record()
        model.finish_gradient_synchronization()
        comm_step_end.record()

        optimizer.step()

        step_end.record()
        torch.cuda.synchronize(device=device)

        step_time_ms = step_start.elapsed_time(step_end)
        comm_time_ms = comm_step_start.elapsed_time(comm_step_end)

        if step >= warmup_steps:
            step_times.append(step_time_ms)
            comm_times.append(comm_time_ms)

    dist.barrier()

    if rank == 0:
        avg_step_time_ms = sum(step_times) / len(step_times)
        avg_comm_time_ms = sum(comm_times) / len(comm_times)
        avg_comm_proportion = (avg_comm_time_ms / avg_step_time_ms) * 100 if avg_step_time_ms > 0 else 0

        print("\n=== Timing Summary (Overlapping DDP) ===")
        print(f"Bucket Size: {args.bucket_size_mb} MB")
        print(f"Average Step Time: {avg_step_time_ms:.2f} ms")
        print(f"Average Comm Time: {avg_comm_time_ms:.2f} ms")
        print(f"Average Comm Proportion: {avg_comm_proportion:.2f}%")

    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket_size_mb", type=float, required=True)
    args = parser.parse_args()
    mp.spawn(main,
             args=(2, args),
             nprocs=2,
             join=True)