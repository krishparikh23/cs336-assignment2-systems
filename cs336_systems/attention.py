import torch
import triton
import triton.language as tl
from einops import rearrange
import math

class FlashAttentionPyTorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        Q, K, V = Q.contiguous(), K.contiguous(), V.contiguous()

        device = Q.device

        B, N_q, D = Q.shape
        _, N_k, Dk = K.shape
        _, _, Dv = V.shape

        dtype_acc = torch.float32
        dtype_out = Q.dtype

        scale = D ** -0.5

        Q_TILE_SIZE = 64
        K_TILE_SIZE = 64
        Q_TILE_SIZE = max(16, Q_TILE_SIZE)
        K_TILE_SIZE = max(16, K_TILE_SIZE)

        Tq = math.ceil(N_q / Q_TILE_SIZE)
        Tk = math.ceil(N_k / K_TILE_SIZE)

        O = torch.zeros_like(Q, dtype=dtype_out)
        L = torch.zeros((B, N_q), dtype=dtype_acc, device=device)

        for i in range(Tq):
            q_start = i * Q_TILE_SIZE
            q_end = min((i + 1) * Q_TILE_SIZE, N_q)
            q_block_size = q_end - q_start
            q_i = Q[:, q_start:q_end, :]

            o_i = torch.zeros((B, q_block_size, D), dtype=dtype_acc, device=device)
            l_i = torch.zeros((B, q_block_size), dtype=dtype_acc, device=device)
            m_i = torch.full((B, q_block_size), float('-inf'), dtype=dtype_acc, device=device)

            for j in range(Tk):
                k_start = j * K_TILE_SIZE
                k_end = min((j + 1) * K_TILE_SIZE, N_k)
                k_j = K[:, k_start:k_end, :]
                v_j = V[:, k_start:k_end, :]

                q_i_acc = q_i.to(dtype_acc)
                k_j_acc = k_j.to(dtype_acc)
                v_j_acc = v_j.to(dtype_acc)

                s_ij = (q_i_acc @ rearrange(k_j_acc, 'b k d -> b d k')) * scale


                m_ij, _ = torch.max(s_ij, dim=-1)
                m_i_prev = m_i
                m_i = torch.maximum(m_i_prev, m_ij)

                p_tilde_ij = torch.exp(s_ij - rearrange(m_i.detach(), 'b q -> b q 1'))

                l_scale = torch.exp(m_i_prev.detach() - m_i.detach())
                p_rowsum = torch.sum(p_tilde_ij, dim=-1)
                l_i = l_scale * l_i + p_rowsum

                o_scale = rearrange(l_scale, 'b q -> b q 1')
                p_dot_v = p_tilde_ij @ v_j_acc
                o_i = o_scale * o_i + p_dot_v

            l_i_safe = l_i + 1e-6

            o_i_final = o_i * rearrange(1.0 / l_i_safe, 'b q -> b q 1')

            L_i_final = m_i + torch.log(l_i_safe)

            O[:, q_start:q_end, :] = o_i_final.to(dtype_out)
            L[:, q_start:q_end] = L_i_final

        ctx.save_for_backward(Q, K, V, O, L)
        ctx.scale = scale
        ctx.is_causal = is_causal

        return O

    @staticmethod
    def backward(ctx, grad_o):
        grad_o = grad_o.contiguous()

        Q, K, V, O, L = [t.contiguous() for t in ctx.saved_tensors]
        scale = ctx.scale

        dtype_out = Q.dtype
        dtype_acc = torch.float32

        Q_acc = Q.to(dtype_acc)
        K_acc = K.to(dtype_acc)
        V_acc = V.to(dtype_acc)
        O_acc = O.to(dtype_acc)
        L_acc = L.to(dtype_acc)
        grad_o_acc = grad_o.to(dtype_acc)

        B, N_q, D = Q_acc.shape

        D_val = torch.sum(O_acc * grad_o_acc, dim=2, keepdim=True)

        s = (Q_acc @ rearrange(K_acc, 'b nk d -> b d nk')) * scale


        p = torch.exp(s - rearrange(L_acc, 'b nq -> b nq 1'))

        grad_v_acc = rearrange(p, 'b nq nk -> b nk nq') @ grad_o_acc

        dp = grad_o_acc @ rearrange(V_acc, 'b nk d -> b d nk')

        ds = p * (dp - D_val)

        grad_q_acc = (ds @ K_acc) * scale

        grad_k_acc = (rearrange(ds, 'b nq nk -> b nk nq') @ Q_acc) * scale

        grad_q = grad_q_acc.to(dtype_out)
        grad_k = grad_k_acc.to(dtype_out)
        grad_v = grad_v_acc.to(dtype_out)

        return grad_q, grad_k, grad_v, None

@triton.jit
def flash_attention_fwd(
    Q_ptr, K_ptr, V_ptr,
    O_ptr,
    L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    is_causal: tl.constexpr,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_batch_ptr = Q_ptr + batch_index * stride_qb
    K_batch_ptr = K_ptr + batch_index * stride_kb
    V_batch_ptr = V_ptr + batch_index * stride_vb
    O_batch_ptr = O_ptr + batch_index * stride_ob
    L_batch_ptr = L_ptr + batch_index * stride_lb

    q_start_row = query_tile_index * Q_TILE_SIZE

    Q_block_ptr = tl.make_block_ptr(
        base=Q_batch_ptr, shape=(N_QUERIES, D), strides=(stride_qq, stride_qd),
        offsets=(q_start_row, 0), block_shape=(Q_TILE_SIZE, D), order=(1, 0)
    )
    K_block_ptr = tl.make_block_ptr(
        base=K_batch_ptr, shape=(N_KEYS, D), strides=(stride_kk, stride_kd),
        offsets=(0, 0), block_shape=(K_TILE_SIZE, D), order=(1, 0)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V_batch_ptr, shape=(N_KEYS, D), strides=(stride_vk, stride_vd),
        offsets=(0, 0), block_shape=(K_TILE_SIZE, D), order=(1, 0)
    )

    q_tile = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    q_tile = q_tile * scale

    o_accumulator = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    l_accumulator = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    m_accumulator = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)

    num_k_tiles = tl.cdiv(N_KEYS, K_TILE_SIZE)
    for k_tile_index in range(num_k_tiles):
        k_start_row = k_tile_index * K_TILE_SIZE

        k_tile = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        v_tile = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

        q_tile_fp32 = q_tile.to(tl.float32)
        k_tile_fp32 = k_tile.to(tl.float32)
        scores = tl.dot(q_tile_fp32, tl.trans(k_tile_fp32))

        if is_causal:
            q_rows = q_start_row + tl.arange(0, Q_TILE_SIZE)
            k_cols = k_start_row + tl.arange(0, K_TILE_SIZE)
            causal_mask = (q_rows[:, None] >= k_cols[None, :])
            scores = tl.where(causal_mask, scores, -1e9)

        current_row_max = tl.max(scores, axis=1)
        m_prev = m_accumulator
        m_new = tl.maximum(m_prev, current_row_max)

        stable_scores = tl.exp(scores - m_new[:, None])

        l_scale_factor = tl.exp(m_prev - m_new)
        l_prev = l_accumulator
        l_accumulator = l_scale_factor * l_prev + tl.sum(stable_scores, axis=1)

        o_accumulator = o_accumulator * l_scale_factor[:, None]
        stable_scores = stable_scores.to(v_tile.dtype)
        o_accumulator = o_accumulator + tl.dot(stable_scores, v_tile)

        m_accumulator = m_new

        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

    l_accumulator = tl.where(l_accumulator == 0.0, 1.0, l_accumulator)
    l_inverse = 1.0 / l_accumulator
    o_final_tile = o_accumulator * l_inverse[:, None]

    logsumexp_tile = m_accumulator + tl.log(l_accumulator)

    O_output_block_ptr = tl.make_block_ptr(
        base=O_batch_ptr, shape=(N_QUERIES, D), strides=(stride_oq, stride_od),
        offsets=(q_start_row, 0), block_shape=(Q_TILE_SIZE, D), order=(1, 0)
    )
    L_output_block_ptr = tl.make_block_ptr(
        base=L_batch_ptr, shape=(N_QUERIES,), strides=(stride_lq,),
        offsets=(q_start_row,), block_shape=(Q_TILE_SIZE,), order=(0,)
    )

    output_dtype = O_ptr.type.element_ty
    o_final_tile = o_final_tile.to(output_dtype)
    tl.store(O_output_block_ptr, o_final_tile, boundary_check=(0, 1))
    tl.store(L_output_block_ptr, logsumexp_tile, boundary_check=(0,))

@torch.compile()
def flash_attention_backward(q, k, v, o, L, grad_o, scale, is_causal):
    q, k, v, o, L, grad_o = [t.contiguous() for t in (q, k, v, o, L, grad_o)]

    dtype_out = q.dtype
    q, k, v, o, grad_o = q.float(), k.float(), v.float(), o.float(), grad_o.float()
    L = L.float()

    B, N_q, D = q.shape
    _, N_k, _ = k.shape

    D_val = torch.sum(o * grad_o, dim=2, keepdim=True)

    s = (q @ rearrange(k, 'b nk d -> b d nk')) * scale

    if is_causal:
        causal_mask = torch.tril(torch.ones((N_q, N_k), device=q.device, dtype=torch.bool)).unsqueeze(0)
        causal_mask = causal_mask.expand(B, -1, -1)
        s = torch.where(causal_mask, s, torch.tensor(-1e9, dtype=s.dtype, device=s.device))

    p = torch.exp(s - rearrange(L, 'b nq -> b nq 1'))

    grad_v = rearrange(p, 'b nq nk -> b nk nq') @ grad_o

    dp = grad_o @ rearrange(v, 'b nk d -> b d nk')

    ds = p * (dp - D_val)

    grad_q = (ds @ k) * scale

    grad_k = (rearrange(ds, 'b nq nk -> b nk nq') @ q) * scale

    grad_q = grad_q.to(dtype_out)
    grad_k = grad_k.to(dtype_out)
    grad_v = grad_v.to(dtype_out)

    return grad_q, grad_k, grad_v


class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal=False, q_tile_size=64, k_tile_size=64):
        batch_size, n_queries, d = q.shape
        _, n_keys, _ = k.shape

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        q_tile_size = max(16, triton.next_power_of_2(q_tile_size))
        k_tile_size = max(16, triton.next_power_of_2(k_tile_size))

        scale = d ** -0.5

        o = torch.empty_like(q)

        L = torch.empty((batch_size, n_queries), device=q.device, dtype=torch.float32)

        grid = ((n_queries + q_tile_size - 1) // q_tile_size, batch_size)

        flash_attention_fwd[grid](
            q, k, v, o, L,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            L.stride(0), L.stride(1),
            n_queries, n_keys,
            scale,
            is_causal=is_causal,
            D=d,
            Q_TILE_SIZE=q_tile_size,
            K_TILE_SIZE=k_tile_size,
        )

        ctx.save_for_backward(q, k, v, o, L)
        ctx.is_causal = is_causal
        ctx.scale = scale

        return o

    @staticmethod
    def backward(ctx, grad_o):
        grad_o = grad_o.contiguous()

        q, k, v, o, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        scale = ctx.scale

        grad_q, grad_k, grad_v = flash_attention_backward(q, k, v, o, L, grad_o, scale, is_causal)

        return grad_q, grad_k, grad_v, None, None, None