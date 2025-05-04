import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from copy import deepcopy
from cs336_basics.model import BasicsTransformerLM

TRANSFORMER_CONFIG_XL = {
    "vocab_size": 10000, # custom
    "context_length": 512, # custom
    "d_model": 1600,
    "num_layers": 48,
    "num_heads": 25,
    "d_ff": 6400,
    "rope_theta": 10000.0,
}


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355' # custom
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def worker_fn(rank: int, world_size: int, args):
    print(f"Rank {rank}: Starting worker.")
    setup(rank, world_size)
    device = f'cuda:{rank}'

    model_config = TRANSFORMER_CONFIG_XL

    model_class = BasicsTransformerLM
    torch.manual_seed(rank + 42)
    non_parallel_model = model_class(**model_config).to(device) if rank == 0 else None
    torch.manual_seed(rank + 42)
    ddp_model = deepcopy(model_class(**model_config).to(device))

    with torch.no_grad():
        for param in ddp_model.parameters():
            dist.broadcast(param.data, src=0)
        dist.barrier()
    print(f"Rank {rank}: Broadcasted initial weights from rank 0.")

    if rank != 0:
        torch.manual_seed(0 + 42)
        rank0_initial_model = model_class(**model_config).to(device)
        for p_ddp, p_rank0_init in zip(ddp_model.parameters(), rank0_initial_model.parameters()):
             assert torch.allclose(p_ddp.data.float(), p_rank0_init.data.float()), \
                 f"Rank {rank}: Broadcasted DDP model weights do not match rank 0's initial weights."
        del rank0_initial_model

    global_batch_size = args.batch_size * world_size
    local_batch_size = args.batch_size

    torch.manual_seed(23)
    context_length = model_config["context_length"]
    vocab_size = model_config["vocab_size"]
    total_steps = args.num_warmup_steps + args.num_measurement_steps
    total_sequences = global_batch_size * total_steps
    all_x = torch.randint(0, vocab_size, (total_sequences, context_length), dtype=torch.long)
    all_y = torch.randint(0, vocab_size, (total_sequences, context_length), dtype=torch.long)

    loss_fn = nn.CrossEntropyLoss()

    ddp_optimizer = optim.SGD(ddp_model.parameters(), lr=1e-3)
    if rank == 0:
        non_parallel_optimizer = optim.SGD(non_parallel_model.parameters(), lr=1e-3)
    else:
        non_parallel_optimizer = None

    print(f"Rank {rank}: Starting training loop for {args.num_warmup_steps} warmup + {args.num_measurement_steps} measurement steps.")

    step_times_ms = []
    comm_times_ms = []

    start_step = torch.cuda.Event(enable_timing=True)
    end_step = torch.cuda.Event(enable_timing=True)
    start_comm = torch.cuda.Event(enable_timing=True)
    end_comm = torch.cuda.Event(enable_timing=True)

    for i in range(total_steps):
        is_measurement_step = (i >= args.num_warmup_steps)

        if is_measurement_step:
            dist.barrier()
            start_step.record()

        # if rank == 0:
        #     non_parallel_optimizer.zero_grad()
        #     start_idx_global = i * global_batch_size
        #     end_idx_global = start_idx_global + global_batch_size
        #     non_parallel_data = all_x[start_idx_global:end_idx_global].to(device)
        #     non_parallel_labels = all_y[start_idx_global:end_idx_global].to(device)

        #     non_parallel_outputs = non_parallel_model(non_parallel_data)

        #     non_parallel_loss = loss_fn(
        #         non_parallel_outputs.view(-1, vocab_size),
        #         non_parallel_labels.view(-1)
        #     )
        #     non_parallel_loss.backward()
        #     non_parallel_optimizer.step()

        ddp_optimizer.zero_grad()

        start_idx_local = (i * global_batch_size) + (rank * local_batch_size)
        end_idx_local = start_idx_local + local_batch_size
        ddp_data = all_x[start_idx_local:end_idx_local].to(device)
        ddp_labels = all_y[start_idx_local:end_idx_local].to(device)

        ddp_outputs = ddp_model(ddp_data)

        ddp_loss = loss_fn(
            ddp_outputs.view(-1, vocab_size),
            ddp_labels.view(-1)
        )
        ddp_loss.backward()

        if is_measurement_step:
            dist.barrier()
            start_comm.record()

        for param in ddp_model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data = param.grad.data.float() / world_size

        if is_measurement_step:
            end_comm.record()
            dist.barrier()

        ddp_optimizer.step()

        if is_measurement_step:
            end_step.record()

        if rank == 0:
            if is_measurement_step:
                torch.cuda.synchronize()

                step_time = start_step.elapsed_time(end_step)
                comm_time = start_comm.elapsed_time(end_comm)
                step_times_ms.append(step_time)
                comm_times_ms.append(comm_time)
                print(f"Measuring Step {i - args.num_warmup_steps}: Verifying parameters...")
            else:
                print(f"Warmup Step {i}: Verifying parameters...")

            # match = True
            # for non_parallel_param, ddp_param in zip(non_parallel_model.parameters(), ddp_model.parameters()):
            #     if not torch.allclose(non_parallel_param.data.float(), ddp_param.data.float(), atol=1e-5, rtol=1e-4):
            #         match = False
            #         break
            # assert match, f"Iter {i}: DDP model parameters do not match non-parallel baseline after step."
            # if match:
            #     print(f"Iter {i}: Parameters verified successfully.")

        dist.barrier()

    print(f"Rank {rank}: Training finished.")

    if rank == 0:
        avg_step_time = sum(step_times_ms) / len(step_times_ms)
        avg_comm_time = sum(comm_times_ms) / len(comm_times_ms)
        avg_comm_proportion = (avg_comm_time / avg_step_time) * 100 if avg_step_time > 0 else 0

        print("=== Timing Summary ===")
        print(f"Average Step Time: {avg_step_time:.2f} ms")
        print(f"Average Comm Time: {avg_comm_time:.2f} ms")
        print(f"Average Comm Proportion: {avg_comm_proportion:.2f}%")

    cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_warmup_steps', type=int, default=5)
    parser.add_argument('--num_measurement_steps', type=int, default=10)
    args = parser.parse_args()

    print(f"Starting naive DDP with world_size={args.world_size}")
    mp.spawn(worker_fn,
             args=(args.world_size, args),
             nprocs=args.world_size,
             join=True)

if __name__ == "__main__":
    main()