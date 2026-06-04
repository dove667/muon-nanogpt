
import numpy as np
import torch
import torch.distributed as dist


@torch.no_grad
def sparse_comms_start(idxes_np, N, rank, world, send_idxes_buffer, *, device):
    rows_per_rank = N // world

    send_idxes = send_idxes_buffer[: idxes_np.shape[0]]
    send_idxes.copy_(torch.from_numpy(idxes_np))
    send_idxes = send_idxes.to(device, non_blocking=True)

    insertion_points = np.searchsorted(
        idxes_np,
        np.arange(0, rows_per_rank * (world + 1), rows_per_rank, dtype=np.int32),
    )
    send_counts = torch.from_numpy(insertion_points[1:] - insertion_points[:-1])
    send_counts[rank] = 0

    send_idxes = torch.cat(
        [send_idxes[: insertion_points[rank]], send_idxes[insertion_points[rank + 1] :]]
    )

    recv_counts = torch.empty_like(send_counts)
    recv_counts_fut = dist.all_to_all_single(recv_counts, send_counts, async_op=True).get_future()
    return send_idxes, send_counts, recv_counts, recv_counts_fut


@torch.no_grad
def sparse_comms_share_indexes(send_idxes, send_counts, recv_counts, *, device):
    total_recv_count = recv_counts.sum().item()
    recv_counts = recv_counts.tolist()
    send_counts = send_counts.tolist()

    recv_idxes = torch.empty(total_recv_count, dtype=torch.int32, device=device)
    idxes_fut = dist.all_to_all_single(
        recv_idxes,
        send_idxes,
        output_split_sizes=recv_counts,
        input_split_sizes=send_counts,
        async_op=True,
    ).get_future()

    sparse_state = {
        "send_idxes": send_idxes,
        "send_counts": send_counts,
        "recv_counts": recv_counts,
    }
    return recv_idxes, sparse_state, idxes_fut


@torch.no_grad
def sparse_comms_share_gradients(grad, idxes, send_counts, recv_counts):
    send_vals = grad[idxes]
    d = grad.shape[1]

    send_sizes = [count * d for count in send_counts]
    recv_sizes = [count * d for count in recv_counts]
    recv_vals = torch.empty(sum(recv_sizes), device=send_vals.device, dtype=grad.dtype)

    val_fut = dist.all_to_all_single(
        recv_vals,
        send_vals.view(-1),
        input_split_sizes=send_sizes,
        output_split_sizes=recv_sizes,
        async_op=True,
    ).get_future()

    return recv_vals, val_fut


@torch.no_grad
def sparse_comms_merge_gradients(grad, recv_idx, recv_vals, rank, world):
    d = grad.shape[1]
    rows_per_rank = grad.shape[0] // world
    grad.index_add_(0, recv_idx, recv_vals.view(-1, d))
    return grad[rows_per_rank * rank : rows_per_rank * (rank + 1)].mul_(1 / world)
