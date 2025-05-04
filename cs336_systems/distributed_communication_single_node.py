import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import os
import argparse
import pandas as pd
import sys

def setup(rank, world_size, backend, master_addr, master_port):
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port

    print(f"Rank {rank}: Initializing process group with backend {backend}, world size {world_size} Addr: {master_addr}:{master_port}")
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    print(f"Rank {rank}: Process group initialized successfully.")

def cleanup():
    dist.destroy_process_group()

def worker_fn(rank, world_size, backend, master_addr, master_port):

    if backend == 'nccl':
        device_type = 'cuda'
        worker_device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(worker_device)
    elif backend == 'gloo':
        device_type = 'cpu'
        worker_device = torch.device("cpu")

    print(f"Worker {rank}/{world_size} starting. Backend: {backend}, Device: {device_type} ({worker_device})")

    setup(rank, world_size, backend, master_addr, master_port)


    sizes_mb = [1, 10, 100, 1024]
    warmup_iters = 5
    measure_iters = 10

    results = []

    for size_mb_val in sizes_mb:
        size_bytes_val = size_mb_val * 1024 * 1024
        elements = size_bytes_val // 4

        if rank == 0:
            print(f"Rank {rank}: Processing size {size_mb_val} mb")

        data = torch.randn(elements, dtype=torch.float32, device=worker_device)

        dist.barrier()

        for _ in range(warmup_iters):
            dist.all_reduce(data, op=dist.ReduceOp.SUM, async_op=False)
        if device_type == 'cuda':
            torch.cuda.synchronize(device=worker_device)

        dist.barrier()

        start_time, end_time, total_time = 0.0, 0.0, 0.0
        if device_type == 'cuda':
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(device=worker_device)
            start_event.record()
            for _ in range(measure_iters):
                dist.all_reduce(data, op=dist.ReduceOp.SUM, async_op=False)
            end_event.record()
            torch.cuda.synchronize(device=worker_device)
            total_time = start_event.elapsed_time(end_event)
        else:
            start_time = time.perf_counter()
            for _ in range(measure_iters):
                dist.all_reduce(data, op=dist.ReduceOp.SUM, async_op=False)
            end_time = time.perf_counter()
            total_time = (end_time - start_time) * 1000

        avg_time = total_time / measure_iters

        results.append({
            'rank': rank,
            'backend': backend,
            'device': device_type,
            'size_mb': size_mb_val,
            'avg_time_ms': avg_time
        })
        dist.barrier()


    all_rank_results_list = [None] * world_size
    dist.gather_object(
        results,
        all_rank_results_list if rank == 0 else None,
        dst=0
    )

    if rank == 0:
        print(f"Processing results on Rank 0 for world_size={world_size}, backend={backend}, device={device_type}...")
        flat_results = []
        if all(res is not None for res in all_rank_results_list):
            for rank_results in all_rank_results_list:
                if isinstance(rank_results, list):
                    flat_results.extend(rank_results)


        if not flat_results:
            print("Error: No results collected on Rank 0. Cannot generate report.")
        else:
            df = pd.DataFrame(flat_results)
            df['world_size'] = world_size

            agg_df = df.groupby(['backend', 'device', 'world_size', 'size_mb'])['avg_time_ms'].agg(['mean', 'std']).reset_index()
            agg_df = agg_df.rename(columns={'mean': 'avg_time_ms', 'std': 'std_dev_ms'})
            agg_df['std_dev_ms'] = agg_df['std_dev_ms'].fillna(0.0)

            print(f"--- Aggregated Benchmark Results (Device: {device_type}) ---")
            agg_df['avg_time_ms'] = agg_df['avg_time_ms'].map('{:.4f}'.format)
            agg_df['std_dev_ms'] = agg_df['std_dev_ms'].map('{:.4f}'.format)
            print(agg_df.to_markdown(index=False, floatfmt=".4f"))
            print("-" * 60)

    cleanup()
    print(f"Worker {rank} finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark PyTorch Distributed All-Reduce on Single Node')
    parser.add_argument('--nprocs', type=int, required=True)
    parser.add_argument('--backend', type=str, required=True, choices=['gloo', 'nccl'])
    parser.add_argument('--master_addr', type=str, default='127.0.0.1')
    parser.add_argument('--master_port', type=str, default='12355')

    args = parser.parse_args()

    device_type = 'cuda' if args.backend == 'nccl' else 'cpu'

    print(f"Starting benchmark with {args.nprocs} processes. Backend: {args.backend}, Device: {device_type.upper()}")


    mp.spawn(
        worker_fn,
        args=(args.nprocs, args.backend, args.master_addr, args.master_port),
        nprocs=args.nprocs,
        join=True
    )

    print("Main process finished.") 