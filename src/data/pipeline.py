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
        tokens = torch.empty(num_tokens, dtype=torch.uint16)
        handle.seek(256 * 4)
        nbytes = handle.readinto(tokens.numpy())
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens


def data_generator(filename_pattern: str, tokens_per_step: int, seq_len: int, grad_accum_steps: int):
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if tokens_per_step % grad_accum_steps != 0:
        raise ValueError("tokens_per_step must be divisible by grad_accum_steps")

    tokens_per_microbatch = tokens_per_step // grad_accum_steps
    if tokens_per_microbatch % seq_len != 0:
        raise ValueError("tokens_per_microbatch must be divisible by seq_len")
    sequences_per_microbatch = tokens_per_microbatch // seq_len

    files = [Path(file) for file in sorted(glob.glob(filename_pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {filename_pattern}")
    shard_idx = 0
    pos = 0
    current_tokens = load_data_shard(files[shard_idx])

    def next_token_block(block_len: int) -> Tensor:
        nonlocal shard_idx, pos, current_tokens
        parts: list[Tensor] = []
        remaining = block_len
        while remaining > 0:
            available = current_tokens.numel() - pos
            take = min(remaining, available)
            parts.append(current_tokens[pos:pos + take])
            pos += take
            remaining -= take
            if pos >= current_tokens.numel():
                shard_idx = (shard_idx + 1) % len(files)
                pos = 0
                current_tokens = load_data_shard(files[shard_idx])
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)

    while True:
        buf = next_token_block(tokens_per_microbatch + sequences_per_microbatch)
        inputs = buf[:-sequences_per_microbatch].view(sequences_per_microbatch, seq_len).to(dtype=torch.int64)
        targets = buf[1:].view(sequences_per_microbatch, seq_len).to(dtype=torch.int64)
        yield (
            inputs.pin_memory().to(device="cuda", non_blocking=True),
            targets.pin_memory().to(device="cuda", non_blocking=True),
        )
