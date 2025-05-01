# attention.py

import torch
import triton
import triton.language as tl
import functools # Import functools for torch.compile
import math # Add math import for ceiling division

# Helper for ceiling division
cdiv = triton.cdiv

def cdiv_py(a, b):
  """Ceiling division for Python integers."""
  return (a + b - 1) // b

@triton.jit
def flash_fwd_kernel(
    # Pointers to Q, K, V matrices
    Q_ptr, K_ptr, V_ptr,
    # Pointer to Output matrix O
    O_ptr,
    # Pointer to Logsumexp matrix L (for backward pass)
    L_ptr,
    # Strides for Q, K, V (Batch, SeqLen, Dim)
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    # Strides for O (Batch, SeqLen, Dim)
    stride_ob, stride_oq, stride_od,
    # Strides for L (Batch, SeqLen)
    stride_lb, stride_lq,
    # Matrix dimensions (Sequence lengths are runtime)
    N_QUERIES, N_KEYS,
    # Scale factor (1 / sqrt(d))
    scale,
    # --- Compile-time constants ---
    is_causal: tl.constexpr, # Flag for causal masking
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    """
    Triton Kernel for FlashAttention-2 Forward Pass (Algorithm 1).
    Includes optional causal masking.

    Computes Attention(Q, K, V) = softmax(Mask(Q @ K.T / sqrt(d))) @ V using tiling.

    Grid: (Tq, batch_size), where Tq = cdiv(N_QUERIES, Q_TILE_SIZE)
    Each program instance corresponds to an iteration of the outer loop (i = 1 to Tq)
    in Algorithm 1, computing one tile of the output matrix O (O_i) and
    the corresponding logsumexp values L (L_i) for a specific batch element.
    """
    # --- Setup ---
    # Get program indices: Each program computes one O_i tile for one batch element.
    query_tile_index = tl.program_id(0)  # Index 'i' for the Q tile (0..Tq-1)
    batch_index = tl.program_id(1)      # Index for batch dimension (0..batch_size-1)

    # Calculate pointer offsets for the current batch element
    Q_batch_ptr = Q_ptr + batch_index * stride_qb
    K_batch_ptr = K_ptr + batch_index * stride_kb
    V_batch_ptr = V_ptr + batch_index * stride_vb
    O_batch_ptr = O_ptr + batch_index * stride_ob
    L_batch_ptr = L_ptr + batch_index * stride_lb

    # Calculate the starting row index for the current Q tile (Q_i)
    # Corresponds to step 1 in Algorithm 1 (conceptually)
    q_start_row = query_tile_index * Q_TILE_SIZE

    # --- Create block pointers for Q, K, V tiles ---
    # These pointers facilitate loading tiles into SRAM.
    # K_block_ptr and V_block_ptr start at the beginning (j=1) of K/V.
    Q_block_ptr = tl.make_block_ptr(
        base=Q_batch_ptr, shape=(N_QUERIES, D), strides=(stride_qq, stride_qd),
        offsets=(q_start_row, 0), block_shape=(Q_TILE_SIZE, D), order=(1, 0)
    )
    K_block_ptr = tl.make_block_ptr(
        base=K_batch_ptr, shape=(N_KEYS, D), strides=(stride_kk, stride_kd),
        offsets=(0, 0), block_shape=(K_TILE_SIZE, D), order=(1, 0) # Starts at K^(1)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V_batch_ptr, shape=(N_KEYS, D), strides=(stride_vk, stride_vd),
        offsets=(0, 0), block_shape=(K_TILE_SIZE, D), order=(1, 0) # Starts at V^(1)
    )

    # --- Algorithm 1: Line 4 ---
    # Load Q_i from global memory (HBM) to SRAM.
    q_tile = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    # Apply scaling factor 1/sqrt(d) to Q_i immediately
    q_tile = q_tile * scale # Renormalize Q to apply scale once

    # --- Algorithm 1: Line 5 ---
    # Initialize accumulators: O_i, l_i, m_i
    # Use float32 for accumulators for numerical stability.
    o_accumulator = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    l_accumulator = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32) # Row sums accumulator
    m_accumulator = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32) # Row max accumulator

    # --- Algorithm 1: Line 6 ---
    # Inner loop over K and V tiles (j = 1 to Tk)
    num_k_tiles = tl.cdiv(N_KEYS, K_TILE_SIZE)
    for k_tile_index in range(num_k_tiles):
        # Starting row index for the current K/V tile
        k_start_row = k_tile_index * K_TILE_SIZE

        # --- Algorithm 1: Line 7 ---
        # Load K^(j) and V^(j) tiles from HBM to SRAM.
        k_tile = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        v_tile = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

        # --- Algorithm 1: Line 8 ---
        # Compute score tile S_i^(j) = Q_i @ (K^(j))^T. Scale already applied to q_tile.
        q_tile_fp32 = q_tile.to(tl.float32)
        k_tile_fp32 = k_tile.to(tl.float32)
        # scores = tl.dot(q_tile_fp32, tl.trans(k_tile_fp32)) * scale # (Q_TILE_SIZE, K_TILE_SIZE)
        scores = tl.dot(q_tile_fp32, tl.trans(k_tile_fp32)) # Scale already applied to q_tile

        # --- Causal Masking (Optional, applied to S_i^(j)) ---
        if is_causal:
            # Create coordinate tensors for the current tile relative to sequence start
            q_rows = q_start_row + tl.arange(0, Q_TILE_SIZE)
            k_cols = k_start_row + tl.arange(0, K_TILE_SIZE)
            # Create mask: query_index >= key_index
            causal_mask = (q_rows[:, None] >= k_cols[None, :]) # Shape (Q_TILE_SIZE, K_TILE_SIZE)
            # Apply mask: set scores to -inf where mask is False (key > query)
            scores = tl.where(causal_mask, scores, -1e9) # Use large negative number

        # --- Algorithm 1: Line 9 ---
        # Update row max: m_i^(j) = max(m_i^(j-1), rowmax(S_i^(j)))
        current_row_max = tl.max(scores, axis=1)
        m_prev = m_accumulator
        m_new = tl.maximum(m_prev, current_row_max)

        # --- Algorithm 1: Line 10 ---
        # Compute numerically stable attention scores: P_tilde_i^(j) = exp(S_i^(j) - m_i^(j))
        # Subtract the new row maximum (m_new) for stability.
        stable_scores = tl.exp(scores - m_new[:, None])

        # --- Algorithm 1: Line 11 ---
        # Update row sums: l_i^(j) = exp(m_i^(j-1) - m_i^(j)) * l_i^(j-1) + rowsum(P_tilde_i^(j))
        l_scale_factor = tl.exp(m_prev - m_new) # Scaling factor for previous accumulator l_i^(j-1)
        l_prev = l_accumulator
        l_accumulator = l_scale_factor * l_prev + tl.sum(stable_scores, axis=1)

        # --- Algorithm 1: Line 12 ---
        # Update output accumulator: O_i^(j) = diag(exp(m_i^(j-1) - m_i^(j))) * O_i^(j-1) + P_tilde_i^(j) @ V^(j)
        # Rescale the previous output accumulator O_i^(j-1) before adding the new contribution.
        o_accumulator = o_accumulator * l_scale_factor[:, None]
        # Cast stable scores (P_tilde) to the dtype of V for the matmul.
        stable_scores = stable_scores.to(v_tile.dtype)
        o_accumulator = o_accumulator + tl.dot(stable_scores, v_tile)

        # Update m_i accumulator for the next iteration (m_i^(j-1) becomes m_i^(j))
        m_accumulator = m_new

        # Advance K and V block pointers to the next tile (j -> j+1)
        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

    # --- End of Algorithm 1: Line 6-13 (Inner loop) ---

    # --- Algorithm 1: Line 14 ---
    # Rescale final output tile: O_i = diag(l_i^(Tk))^(-1) * O_i^(Tk)
    # Use the final row sum accumulator l_accumulator = l_i^(Tk).
    # Avoid division by zero for rows that might have had all '-inf' scores (e.g., masked rows).
    l_accumulator = tl.where(l_accumulator == 0.0, 1.0, l_accumulator) # Replace 0 with 1 for division
    l_inverse = 1.0 / l_accumulator
    o_final_tile = o_accumulator * l_inverse[:, None] # Apply scaling

    # --- Algorithm 1: Line 15 ---
    # Compute Logsumexp for backward pass: L_i = m_i^(Tk) + log(l_i^(Tk))
    # Use the final row max (m_accumulator) and row sum (l_accumulator) accumulators.
    # Handle cases where l_accumulator was 0 (and set to 1 above): log(1) = 0.
    # For fully masked rows, m_accumulator is -inf, l_accumulator is 1 -> L_i = -inf.
    logsumexp_tile = m_accumulator + tl.log(l_accumulator)

    # --- Output Block Pointers ---
    # Create block pointers for writing the final O_i and L_i tiles back to HBM.
    O_output_block_ptr = tl.make_block_ptr(
        base=O_batch_ptr, shape=(N_QUERIES, D), strides=(stride_oq, stride_od),
        offsets=(q_start_row, 0), block_shape=(Q_TILE_SIZE, D), order=(1, 0)
    )
    L_output_block_ptr = tl.make_block_ptr(
        base=L_batch_ptr, shape=(N_QUERIES,), strides=(stride_lq,),
        offsets=(q_start_row,), block_shape=(Q_TILE_SIZE,), order=(0,)
    )

    # --- Algorithm 1: Line 16 & 17 ---
    # Write results (O_i, L_i) to global memory (HBM).
    # Cast final output O_i back to the original input data type.
    output_dtype = O_ptr.type.element_ty
    o_final_tile = o_final_tile.to(output_dtype)
    tl.store(O_output_block_ptr, o_final_tile, boundary_check=(0, 1))
    tl.store(L_output_block_ptr, logsumexp_tile, boundary_check=(0,))
    # --- End of Algorithm 1 ---


# --- Compiled PyTorch Backward Pass ---
# Define the backward pass function outside the autograd class
# Apply torch.compile to this function
@torch.compile() # You can add options like mode="reduce-overhead" if needed
def _flash_backward_py(q, k, v, o, L, grad_o, scale, is_causal):
    """
    Computes gradients dQ, dK, dV using PyTorch based on FlashAttention-2 paper eq. 13-19.
    Designed to be compiled with torch.compile. Uses recomputation.

    Args:
        q: Query tensor (B, N_q, D) from forward.
        k: Key tensor (B, N_k, D) from forward.
        v: Value tensor (B, N_k, D) from forward.
        o: Output tensor (B, N_q, D) from forward.
        L: Logsumexp tensor (B, N_q) from forward.
        grad_o: Gradient w.r.t. output O (B, N_q, D).
        scale: Scaling factor (1 / sqrt(D)).
        is_causal: Boolean indicating if causal masking was used in forward.

    Returns:
        Tuple (grad_q, grad_k, grad_v)
    """
    # Ensure inputs are contiguous for performance, especially with torch.compile
    q, k, v, o, L, grad_o = [t.contiguous() for t in (q, k, v, o, L, grad_o)]

    # Use float32 for intermediate computations for stability.
    dtype_out = q.dtype # Output gradients should match original input dtype
    q, k, v, o, grad_o = q.float(), k.float(), v.float(), o.float(), grad_o.float()
    L = L.float() # L should already be float32

    B, N_q, D = q.shape
    _, N_k, _ = k.shape

    # --- Pre-computation ---
    # D = rowsum(O * dO) (tensor D_i in the paper's notation)
    # This is equivalent to rowsum(P * dP). Shape: (B, N_q, 1)
    D_val = torch.sum(o * grad_o, dim=2, keepdim=True)

    # --- Recomputation and Gradient Calculation ---
    # Eq. (13): S = QK^T / sqrt(d)
    # Recompute attention scores S.
    s = (q @ k.transpose(-2, -1)) * scale # Shape: (B, N_q, N_k)

    # Apply causal mask *during recomputation* if it was used in forward.
    # This ensures the recomputed P (next step) is consistent with the forward pass.
    if is_causal:
        causal_mask = torch.tril(torch.ones((N_q, N_k), device=q.device, dtype=torch.bool)).unsqueeze(0) # Shape: (1, N_q, N_k)
        causal_mask = causal_mask.expand(B, -1, -1) # Shape: (B, N_q, N_k)
        # Use a large negative number instead of -inf for numerical stability.
        s = torch.where(causal_mask, s, torch.tensor(-1e9, dtype=s.dtype, device=s.device))

    # Eq. (14): P_ij = exp(S_ij - L_i)
    # Recompute attention probabilities P using the scores S and saved logsumexp L.
    # L needs unsqueezing: (B, N_q) -> (B, N_q, 1) for broadcasting.
    p = torch.exp(s - L.unsqueeze(dim=2)) # Shape: (B, N_q, N_k)

    # Eq. (15): dV = P^T @ dO
    # Gradient w.r.t. V. Shape: (B, N_k, D)
    grad_v = p.transpose(-2, -1) @ grad_o

    # Eq. (16): dP = dO @ V^T
    # Gradient w.r.t P (intermediate). Shape: (B, N_q, N_k)
    dp = grad_o @ v.transpose(-2, -1)

    # Eq. (17): dS_ij = P_ij * (dP_ij - D_i)
    # Gradient w.r.t S (intermediate). D_val needs broadcasting from (B, N_q, 1).
    ds = p * (dp - D_val) # Shape: (B, N_q, N_k)

    # Eq. (18): dQ = (dS @ K) / sqrt(d)
    # Gradient w.r.t. Q. Shape: (B, N_q, D)
    grad_q = (ds @ k) * scale

    # Eq. (19): dK = (dS^T @ Q) / sqrt(d)
    # Gradient w.r.t. K. Shape: (B, N_k, D)
    grad_k = (ds.transpose(-2, -1) @ q) * scale

    # Cast gradients back to original input dtype if needed
    grad_q = grad_q.to(dtype_out)
    grad_k = grad_k.to(dtype_out)
    grad_v = grad_v.to(dtype_out)

    return grad_q, grad_k, grad_v


# --- PyTorch Autograd Function ---

class FlashAttentionTritonFunc(torch.autograd.Function):
    """
    PyTorch autograd function calling the Triton kernel for FlashAttention-2 forward pass
    and a compiled PyTorch function for the backward pass.
    """
    @staticmethod
    # Add is_causal argument here
    def forward(ctx, q, k, v, is_causal=False, q_tile_size=64, k_tile_size=64):
        """
        Forward pass of FlashAttention using the Triton kernel.

        Args:
            ctx: Context object for autograd.
            q: Query tensor (Batch, N_Queries, Dim).
            k: Key tensor (Batch, N_Keys, Dim).
            v: Value tensor (Batch, N_Keys, Dim).
            is_causal (bool): If True, apply causal attention masking.
            q_tile_size: Tile size for Q sequence dimension (Bq).
            k_tile_size: Tile size for K/V sequence dimension (Bk).

        Returns:
            Output tensor (Batch, N_Queries, Dim).
        """
        # Input validation
        assert q.is_cuda and k.is_cuda and v.is_cuda, "Input tensors must be on CUDA device"
        assert q.dtype == k.dtype == v.dtype, "Input tensors must have the same dtype"
        # Contiguous checks are good practice but can sometimes be relaxed depending on kernel
        # assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous(), "Input tensors must be contiguous"

        batch_size, n_queries, d = q.shape
        _, n_keys, d_k = k.shape
        _, _, d_v = v.shape
        assert k.shape[0] == v.shape[0] == batch_size, "Batch size mismatch"
        assert k.shape[1] == v.shape[1] == n_keys, "Key/Value sequence length mismatch"
        assert d_k == d and d_v == d, f"Head dimension mismatch: Q({d}), K({d_k}), V({d_v})"
        # Causal masking typically requires Q and K sequence lengths to be the same
        if is_causal:
            assert n_queries == n_keys, f"Causal attention requires N_Queries ({n_queries}) == N_Keys ({n_keys})"

        # Ensure contiguous inputs for the Triton kernel
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        # Use powers of 2 for tile sizes
        q_tile_size = max(16, triton.next_power_of_2(q_tile_size))
        k_tile_size = max(16, triton.next_power_of_2(k_tile_size))

        # Scale factor
        scale = d ** -0.5

        # Allocate output tensor O
        o = torch.empty_like(q)

        # Allocate logsumexp tensor L (always float32 for stability)
        L = torch.empty((batch_size, n_queries), device=q.device, dtype=torch.float32)

        # Define the launch grid
        grid = (triton.cdiv(n_queries, q_tile_size), batch_size)

        # Launch the Triton kernel
        flash_fwd_kernel[grid](
            # Pointers (runtime)
            q, k, v, o, L,
            # Strides (runtime)
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            L.stride(0), L.stride(1),
            # Dimensions (runtime)
            n_queries, n_keys,
            # Scale (runtime)
            scale,
            # --- Constexpr arguments passed by keyword ---
            is_causal=is_causal, # Pass the flag here
            D=d,
            Q_TILE_SIZE=q_tile_size,
            K_TILE_SIZE=k_tile_size,
            # Optional: Define NUM_WARPS, NUM_STAGES if needed for performance tuning
            # NUM_WARPS=4,
            # NUM_STAGES=2,
        )

        # Save tensors and parameters needed for backward pass
        # O is also needed for D computation
        ctx.save_for_backward(q, k, v, o, L)
        ctx.is_causal = is_causal # Save is_causal flag
        # ctx.q_tile_size = q_tile_size # Not needed for PyTorch backward
        # ctx.k_tile_size = k_tile_size # Not needed for PyTorch backward
        ctx.scale = scale

        return o

    @staticmethod
    def backward(ctx, grad_o):
        """
        Backward pass calling the compiled PyTorch implementation.
        """
        # Ensure grad_o is contiguous before passing to compiled function
        grad_o = grad_o.contiguous()

        # Retrieve saved tensors and parameters
        q, k, v, o, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        scale = ctx.scale

        # Call the compiled PyTorch backward function
        grad_q, grad_k, grad_v = _flash_backward_py(q, k, v, o, L, grad_o, scale, is_causal)

        # Return gradients for q, k, v, and None for non-tensor inputs
        # The order must match the inputs to forward: q, k, v, is_causal, q_tile_size, k_tile_size
        return grad_q, grad_k, grad_v, None, None, None


# --- Pure PyTorch Autograd Function ---
class FlashAttentionPurePyTorch(torch.autograd.Function):
    """
    Implements FlashAttention-2 forward algorithm purely in PyTorch.
    Reference: Algorithm 1 in the FlashAttention-2 paper.
    Does NOT handle causal masking in forward (assumes `is_causal=False`).
    Works on both CPU and CUDA. Includes PyTorch backward pass implementation.
    """
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False): # is_causal is ignored in fwd
        """
        Args:
            ctx: Autograd context.
            Q: Query tensor (Batch, N_Queries, Dim).
            K: Key tensor (Batch, N_Keys, Dim).
            V: Value tensor (Batch, N_Keys, Dim).
            is_causal: Flag for causal masking (currently IGNORED in this forward implementation).

        Returns:
            Output tensor O (Batch, N_Queries, Dim).
        """
        # Ensure inputs are contiguous
        Q, K, V = Q.contiguous(), K.contiguous(), V.contiguous()

        # Device for calculations and output tensors
        device = Q.device

        # Basic shape checks
        B, N_q, D = Q.shape
        _, N_k, Dk = K.shape
        _, _, Dv = V.shape
        assert D == Dk == Dv, f"Dimension mismatch: Q({D}), K({Dk}), V({Dv})"
        assert K.shape[0] == V.shape[0] == B, "Batch size mismatch"
        assert K.shape[1] == V.shape[1] == N_k, "SeqLen mismatch for K/V"

        # Use float32 for accumulators for stability
        dtype_acc = torch.float32
        # Output dtype matches input Q
        dtype_out = Q.dtype

        # Scale factor
        scale = D ** -0.5

        # --- Tile sizes (fixed for simplicity, could be arguments) ---
        # Corresponds to Bq, Bk in Algorithm 1.
        Q_TILE_SIZE = 64
        K_TILE_SIZE = 64
        Q_TILE_SIZE = max(16, Q_TILE_SIZE)
        K_TILE_SIZE = max(16, K_TILE_SIZE)

        # Number of tiles
        Tq = math.ceil(N_q / Q_TILE_SIZE)
        Tk = math.ceil(N_k / K_TILE_SIZE)

        # Output tensors
        O = torch.zeros_like(Q, dtype=dtype_out)
        L = torch.zeros((B, N_q), dtype=dtype_acc, device=device) # Logsumexp accumulator

        # --- Outer loop: Iterate over Q blocks (Algorithm 1: Line 3) ---
        # i corresponds to the outer loop index in Algorithm 1.
        for i in range(Tq):
            # --- Algorithm 1: Line 1 (Implicit Q splitting) & Line 4 ---
            # Determine block boundaries and load Q_i.
            q_start = i * Q_TILE_SIZE
            q_end = min((i + 1) * Q_TILE_SIZE, N_q)
            q_block_size = q_end - q_start
            q_i = Q[:, q_start:q_end, :]

            # --- Algorithm 1: Line 5 ---
            # Initialize accumulators O_i, l_i, m_i for the current Q block.
            o_i = torch.zeros((B, q_block_size, D), dtype=dtype_acc, device=device)
            l_i = torch.zeros((B, q_block_size), dtype=dtype_acc, device=device)
            m_i = torch.full((B, q_block_size), float('-inf'), dtype=dtype_acc, device=device)

            # --- Inner loop: Iterate over K, V blocks (Algorithm 1: Line 6) ---
            # j corresponds to the inner loop index in Algorithm 1.
            for j in range(Tk):
                # --- Algorithm 1: Line 2 (Implicit K/V splitting) & Line 7 ---
                # Determine block boundaries and load K_j, V_j.
                k_start = j * K_TILE_SIZE
                k_end = min((j + 1) * K_TILE_SIZE, N_k)
                k_j = K[:, k_start:k_end, :]
                v_j = V[:, k_start:k_end, :]

                # Cast to accumulator type for computation
                q_i_acc = q_i.to(dtype_acc)
                k_j_acc = k_j.to(dtype_acc)
                v_j_acc = v_j.to(dtype_acc)

                # --- Algorithm 1: Line 8 ---
                # Compute score tile S_ij = (Q_i @ K_j^T) / sqrt(d)
                s_ij = (q_i_acc @ k_j_acc.transpose(-2, -1)) * scale

                # --- Causal Masking (NOT IMPLEMENTED HERE) ---
                # If is_causal=True, masking should be applied to s_ij here.

                # --- Algorithm 1: Line 9 ---
                # Update row max: m_ij = rowmax(S_ij), m_i = max(m_i_prev, m_ij)
                m_ij, _ = torch.max(s_ij, dim=-1)
                m_i_prev = m_i
                m_i = torch.maximum(m_i_prev, m_ij)

                # --- Algorithm 1: Line 10 ---
                # Compute numerically stable attention scores: P_tilde_ij = exp(S_ij - m_i)
                p_tilde_ij = torch.exp(s_ij - m_i.detach().unsqueeze(-1))

                # --- Algorithm 1: Line 11 ---
                # Update row sums: l_i = exp(m_i_prev - m_i) * l_i + rowsum(P_tilde_ij)
                l_scale = torch.exp(m_i_prev.detach() - m_i.detach())
                p_rowsum = torch.sum(p_tilde_ij, dim=-1)
                l_i = l_scale * l_i + p_rowsum

                # --- Algorithm 1: Line 12 ---
                # Update output accumulator: O_i = diag(exp(m_i_prev - m_i)) * O_i + P_tilde_ij @ V_j
                o_scale = l_scale.unsqueeze(-1)
                p_dot_v = p_tilde_ij @ v_j_acc
                o_i = o_scale * o_i + p_dot_v

            # --- End of Inner Loop (Algorithm 1: Line 13) ---

            # --- Post-inner loop updates for the current Q block i ---
            # Add small epsilon for numerical stability before division/log
            l_i_safe = l_i + 1e-6

            # --- Algorithm 1: Line 14 ---
            # Rescale final output tile: O_i = diag(l_i)^(-1) * O_i
            o_i_final = o_i * (1.0 / l_i_safe).unsqueeze(-1)

            # --- Algorithm 1: Line 15 ---
            # Compute logsumexp for backward pass: L_i = m_i + log(l_i)
            L_i_final = m_i + torch.log(l_i_safe)

            # --- Algorithm 1: Line 16 & 17 ---
            # Write results O_i, L_i to their corresponding slices in global O, L tensors.
            O[:, q_start:q_end, :] = o_i_final.to(dtype_out)
            L[:, q_start:q_end] = L_i_final

        # --- End of Outer Loop (Algorithm 1: Line 18) ---

        ctx.save_for_backward(Q, K, V, O, L)
        ctx.scale = scale
        # Store is_causal flag. IMPORTANT: Since fwd ignores it, backward MUST also ignore it
        # during recomputation for consistency.
        ctx.is_causal = is_causal

        # --- Algorithm 1: Line 19 --- Return O (L is implicitly returned via ctx)
        return O

    @staticmethod
    def backward(ctx, grad_o):
        """
        Computes gradients dQ, dK, dV using PyTorch based on FlashAttention-2 paper eq. 13-19.
        Uses recomputation.
        IMPORTANT: Does NOT apply causal masking during recomputation because the
        corresponding forward pass also ignored it. Consistency is key.
        """
        # Ensure grad_o is contiguous
        grad_o = grad_o.contiguous()

        # Retrieve saved tensors and parameters
        # Use contiguous saved tensors for performance
        Q, K, V, O, L = [t.contiguous() for t in ctx.saved_tensors]
        scale = ctx.scale
        # is_causal = ctx.is_causal # Retrieve if needed, but not used in logic below

        # Device and dtypes
        device = Q.device
        dtype_out = Q.dtype
        dtype_acc = torch.float32 # Use float32 for intermediate accumulation

        # Cast inputs needed for intermediate calculations to float32
        Q_acc = Q.to(dtype_acc)
        K_acc = K.to(dtype_acc)
        V_acc = V.to(dtype_acc)
        O_acc = O.to(dtype_acc)
        L_acc = L.to(dtype_acc) # L should already be float32
        grad_o_acc = grad_o.to(dtype_acc)

        B, N_q, D = Q_acc.shape
        _, N_k, _ = K_acc.shape

        # --- Backward Pass Calculations ---

        # Pre-computation D = rowsum(O * dO) (eq. derived before 13)
        # D_i in the paper notation. Shape: (B, N_q, 1)
        D_val = torch.sum(O_acc * grad_o_acc, dim=2, keepdim=True) # Shape: (B, N_q, 1)

        # Eq. (13): Recomputation S = QK^T / sqrt(d)
        s = (Q_acc @ K_acc.transpose(-2, -1)) * scale # Shape: (B, N_q, N_k)

        # --- Causal Mask Handling ---
        # NOTE: Since the forward pass IGNORED is_causal, we MUST also ignore it here
        # during recomputation to be consistent with the computed L.
        # If forward had handled causal masking, we would need to apply it here using ctx.is_causal.
        # Example (if forward had handled masking):
        # if ctx.is_causal:
        #     causal_mask = torch.tril(torch.ones((N_q, N_k), device=device, dtype=torch.bool)).unsqueeze(0)
        #     s = torch.where(causal_mask, s, torch.tensor(-1e9, dtype=s.dtype, device=device))

        # Recomputation Pij = exp(Sij - Li) (eq. 14)
        # L needs unsqueezing: (B, N_q) -> (B, N_q, 1) for broadcasting
        p = torch.exp(s - L_acc.unsqueeze(dim=2)) # Shape: (B, N_q, N_k)

        # Compute dV = P^T dO (eq. 15)
        # (B, N_k, N_q) @ (B, N_q, D) -> (B, N_k, D)
        grad_v_acc = p.transpose(-2, -1) @ grad_o_acc

        # Compute dP = dO V^T (eq. 16)
        # (B, N_q, D) @ (B, D, N_k) -> (B, N_q, N_k)
        dp = grad_o_acc @ V_acc.transpose(-2, -1)

        # Compute dSij = Pij * (dPij - Di) (eq. 17)
        # D_val needs broadcasting from (B, N_q, 1) to (B, N_q, N_k)
        ds = p * (dp - D_val) # Shape: (B, N_q, N_k)

        # Compute dQ = dS K / sqrt(d) (eq. 18)
        # (B, N_q, N_k) @ (B, N_k, D) -> (B, N_q, D)
        grad_q_acc = (ds @ K_acc) * scale

        # Compute dK = dS^T Q / sqrt(d) (eq. 19)
        # (B, N_k, N_q) @ (B, N_q, D) -> (B, N_k, D)
        grad_k_acc = (ds.transpose(-2, -1) @ Q_acc) * scale

        # Cast gradients back to original input dtype
        grad_q = grad_q_acc.to(dtype_out)
        grad_k = grad_k_acc.to(dtype_out)
        grad_v = grad_v_acc.to(dtype_out)

        # Return gradients for Q, K, V, and None for non-tensor input is_causal
        # Order must match the inputs to forward: Q, K, V, is_causal
        return grad_q, grad_k, grad_v, None