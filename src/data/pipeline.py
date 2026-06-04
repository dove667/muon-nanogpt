import glob
from pathlib import Path

import torch
from torch import Tensor


def load_data_shard(file: Path) -> Tensor:
    header = torch.from_file(str(file), False, 256, dtype=torch.int32)
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2])
    with file.open("rb", buffering=0) as handle:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)
        handle.seek(256 * 4)
        nbytes = handle.readinto(tokens.numpy())
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens


def data_generator(filename_pattern: str, num_tokens: int, seq_len: int, grad_accum_steps: int):
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if num_tokens % grad_accum_steps != 0:
        raise ValueError("num_tokens must be divisible by grad_accum_steps")

    microbatch_tokens = num_tokens // grad_accum_steps
    if microbatch_tokens % seq_len != 0:
        raise ValueError("microbatch tokens must be divisible by seq_len")
    batch_size = microbatch_tokens // seq_len

    files = [Path(file) for file in sorted(glob.glob(filename_pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {filename_pattern}")
    shards = [load_data_shard(file) for file in files]
    shard_idx = 0
    pos = 0

    def next_token_block(block_len: int) -> Tensor:
        nonlocal shard_idx, pos
        parts: list[Tensor] = []
        remaining = block_len
        while remaining > 0:
            tokens = shards[shard_idx]
            available = tokens.numel() - pos
            take = min(remaining, available)
            parts.append(tokens[pos:pos + take])
            pos += take
            remaining -= take
            if pos >= tokens.numel():
                shard_idx = (shard_idx + 1) % len(shards)
                pos = 0
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)

    while True:
        buf = next_token_block(microbatch_tokens + batch_size)
        inputs = buf[:-batch_size].view(batch_size, seq_len).to(dtype=torch.int64)
        targets = buf[1:].view(batch_size, seq_len).to(dtype=torch.int64)

        new_params = yield (
            inputs.to(device="cuda", non_blocking=True),
            targets.to(device="cuda", non_blocking=True),
        )

        if new_params is not None:
            new_num_tokens, new_seq_len, new_grad_accum_steps = new_params
            if new_num_tokens % new_grad_accum_steps != 0:
                raise ValueError("updated num_tokens must be divisible by grad_accum_steps")
            microbatch_tokens = new_num_tokens // new_grad_accum_steps
            if microbatch_tokens % new_seq_len != 0:
                raise ValueError("updated microbatch tokens must be divisible by seq_len")
            seq_len = new_seq_len
            batch_size = microbatch_tokens // seq_len
