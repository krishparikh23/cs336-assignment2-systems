import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import os
import argparse
import pandas as pd
import sys # Added sys for exiting

def setup(rank, world_size, backend, master_addr, master_port):
    """Initializes the distributed environment."""
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port

    # Initialize the process group
    print(f"Rank {rank}: Initializing process group with backend {backend}, world size {world_size} Addr: {master_addr}:{master_port}")
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    print(f"Rank {rank}: Process group initialized successfully.")

def cleanup():
    """Destroys the distributed process group."""
    dist.destroy_process_group()

def worker_fn(rank, world_size, backend, master_addr, master_port):
    """Runs the all-reduce benchmark for a given configuration in a spawned process."""

    # Determine device based on backend
    if backend == 'nccl':
        device_type = 'cuda'
        
        worker_device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(worker_device)
    elif backend == 'gloo':
        device_type = 'cpu'
        worker_device = torch.device("cpu")

    print(f"Worker {rank}/{world_size} starting. Backend: {backend}, Device: {device_type} ({worker_device})")

    # Setup connections
    setup(rank, world_size, backend, master_addr, master_port)


    # Benchmark settings
    sizes_mb = [1, 10, 100, 1024] # Define sizes in MB
    warmup_iters = 5
    measure_iters = 10

    results = []

    # Iterate through the defined sizes in MB
    for size_mb_val in sizes_mb:
        # Calculate size in bytes and number of elements inside the loop
        size_bytes_val = size_mb_val * 1024 * 1024
        elements = size_bytes_val // 4 # float32 is 4 bytes

        if rank == 0:
            print(f"Rank {rank}: Processing size {size_mb_val} mb")

        # 'elements' is now computed within the loop for the current size
        data = torch.randn(elements, dtype=torch.float32, device=worker_device)

        dist.barrier() # Ensure all processes created tensor before starting benchmark

        # Warm-up
        for _ in range(warmup_iters):
            dist.all_reduce(data, op=dist.ReduceOp.SUM, async_op=False)
        if device_type == 'cuda':
            torch.cuda.synchronize(device=worker_device) # Sync the specific device

        # Measurement
        dist.barrier() # Sync all processes before starting measurement

        start_time, end_time, total_time = 0.0, 0.0, 0.0
        if device_type == 'cuda':
            # Always use CUDA events now
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(device=worker_device) # Ensure sync before starting timer
            start_event.record()
            for _ in range(measure_iters):
                dist.all_reduce(data, op=dist.ReduceOp.SUM, async_op=False)
            end_event.record()
            torch.cuda.synchronize(device=worker_device) # Wait for all operations to complete
            total_time = start_event.elapsed_time(end_event) # Time in milliseconds
        else: # CPU
            start_time = time.perf_counter()
            for _ in range(measure_iters):
                dist.all_reduce(data, op=dist.ReduceOp.SUM, async_op=False)
            # No explicit sync needed for CPU blocking calls, but barrier ensures completion
            end_time = time.perf_counter()
            total_time = (end_time - start_time) * 1000 # Convert to milliseconds

        avg_time = total_time / measure_iters

        results.append({
            'rank': rank,
            'backend': backend,
            'device': device_type,
            'size_mb': size_mb_val,
            'avg_time_ms': avg_time
        })
        dist.barrier() # Ensure all processes finish this size before starting next


    # Gather results from all ranks to rank 0
    all_rank_results_list = [None] * world_size
    # Use gather_object which works for both CPU and CUDA tensors in the list/dict
    dist.gather_object(
        results,
        all_rank_results_list if rank == 0 else None,
        dst=0
    )

    # Rank 0 processes and prints results
    if rank == 0:
        print(f"Processing results on Rank 0 for world_size={world_size}, backend={backend}, device={device_type}...")
        flat_results = []
        # Check if gather_object actually populated the list
        if all(res is not None for res in all_rank_results_list):
            for rank_results in all_rank_results_list:
                 # Ensure rank_results is iterable (it should be a list of dicts)
                if isinstance(rank_results, list):
                    flat_results.extend(rank_results)
                else:
                    print(f"Warning: Received unexpected data type from rank: {type(rank_results)}. Skipping.")
        else:
             print("Warning: Did not receive results from all ranks.")


        if not flat_results:
            print("Error: No results collected on Rank 0. Cannot generate report.")
        else:
            df = pd.DataFrame(flat_results)
            df['world_size'] = world_size

            # Aggregate results: Calculate mean and std deviation across ranks for this run
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
    parser.add_argument('--nprocs', type=int, required=True, help='Number of processes (world size)')
    parser.add_argument('--backend', type=str, required=True, choices=['gloo', 'nccl'],
                        help='Distributed backend to use (gloo for CPU, nccl for GPU)')
    parser.add_argument('--master_addr', type=str, default='127.0.0.1', help='Master address for distributed setup')
    parser.add_argument('--master_port', type=str, default='12355', help='Master port for distributed setup')

    args = parser.parse_args()

    # Determine device type based on backend
    device_type = 'cuda' if args.backend == 'nccl' else 'cpu'

    print(f"Starting benchmark with {args.nprocs} processes. Backend: {args.backend}, Device: {device_type.upper()}")


    # Use mp.spawn to launch processes
    mp.spawn(
        worker_fn,
        # Arguments to pass to worker_fn (after rank)
        args=(args.nprocs, args.backend, args.master_addr, args.master_port),
        nprocs=args.nprocs,
        join=True
    )

    print("Main process finished.") 

"""
uv run python distributed_communication_single_node.py --nprocs 2 --backend gloo > benchmark_results.txt && uv run python distributed_communication_single_node.py --nprocs 4 --backend gloo >> benchmark_results.txt && uv run python distributed_communication_single_node.py --nprocs 6 --backend gloo >> benchmark_results.txt && uv run python distributed_communication_single_node.py --nprocs 2 --backend nccl >> benchmark_results.txt && uv run python distributed_communication_single_node.py --nprocs 4 --backend nccl >> benchmark_results.txt && uv run python distributed_communication_single_node.py --nprocs 6 --backend nccl >> benchmark_results.txt
"""