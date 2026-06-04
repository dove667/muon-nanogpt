
import glob
import threading
from itertools import cycle
from pathlib import Path

import torch
from torch import Tensor

BOS_ID = 50256
TRAIN_MAX_NUM_DOCS = {16384: 64, 32768: 96, 49152: 128}


def next_multiple_of_n(v: float | int, *, n: int) -> int:
    import math
    return math.ceil(v / n) * n


def load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32)
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2])
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy())
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens


class Shard:
    def __init__(self, tokens: Tensor, world_size: int = 1):
        self.tokens = tokens
        self.size = tokens.numel()
        self.world_size = world_size
        self.i = 0
        self.bos_idx = (tokens[:6_000_000] == BOS_ID).nonzero(as_tuple=True)[0].to(torch.int64).cpu().numpy()
        self._full_idx = None
        self._loader_thread = None
        self._ready = threading.Event()
        self._loader_thread = threading.Thread(target=self._scan)
        self._loader_thread.start()

    def _scan(self):
        self._full_idx = (self.tokens == BOS_ID).nonzero(as_tuple=True)[0].to(torch.int64).cpu().numpy()
        self._ready.set()

    def _maybe_switch(self):
        if self.bos_idx is not self._full_idx and self._ready.is_set():
            self._loader_thread.join()
            self.bos_idx = self._full_idx

    def next_batch(self, num_tokens_local: int, max_seq_len: int):
        self._maybe_switch()
        n = len(self.bos_idx)
        starts = [[] for _ in range(self.world_size)]
        ends = [[] for _ in range(self.world_size)]

        idx = self.i
        for r in range(self.world_size):
            cur_len = 0
            while cur_len <= num_tokens_local:
                if idx >= n:
                    raise StopIteration("Insufficient BOS ahead; hit tail of shard.")
                cur = self.bos_idx[idx]
                starts[r].append(cur)
                idx += 1
                end = min(self.bos_idx[idx] if idx < n else self.size, cur + max_seq_len, cur + num_tokens_local - cur_len + 1)
                ends[r].append(end)
                cur_len += end - cur
            assert cur_len == num_tokens_local + 1
        self.i = idx
        return starts, ends

    @staticmethod
    def load_async(file: Path, world_size: int = 1):
        result = {}
        ready = threading.Event()
        def load():
            tokens = load_data_shard(file)
            result["shard"] = Shard(tokens, world_size)
            ready.set()
        thread = threading.Thread(target=load)
        thread.start()
        def get():
            ready.wait()
            thread.join()
            return result["shard"]
        return get


def get_bigram_hash(x: Tensor, bigram_vocab_size: int):
    rand_int_1 = 36313
    rand_int_2 = 27191
    mod = bigram_vocab_size - 1
    x = x.to(torch.int32)
    out = torch.empty_like(x, pin_memory=True)
    out.copy_(x)
    out[0] = mod
    out[1:] = torch.bitwise_xor(rand_int_1 * out[1:], rand_int_2 * out[:-1]) % mod
    return out


def distributed_data_generator(filename_pattern: str, num_tokens: int, max_seq_len: int, grad_accum_steps: int, align_to_bos: bool, rank: int, world_size: int, bigram_vocab_size: int):
    assert num_tokens % (world_size * grad_accum_steps) == 0, "Batch size must be divisible by world size"
    num_tokens = num_tokens // grad_accum_steps
    files = [Path(file) for file in sorted(glob.glob(filename_pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {filename_pattern}")

    file_iter = cycle(files)
    tokens = load_data_shard(next(file_iter))
    if align_to_bos:
        shard = Shard(tokens, world_size)
        next_shard_getter = Shard.load_async(next(file_iter), world_size)
    else:
        pos = 0

    while True:
        num_tokens_local = num_tokens // world_size
        max_num_docs = TRAIN_MAX_NUM_DOCS.get(num_tokens_local, next_multiple_of_n(num_tokens_local // 300, n=128))

        if align_to_bos:
            try:
                seq_starts, seq_ends = shard.next_batch(num_tokens_local, max_seq_len)
                start_idxs, end_idxs = torch.tensor(seq_starts[rank]), torch.tensor(seq_ends[rank])
            except StopIteration:
                shard = next_shard_getter()
                tokens = shard.tokens
                try:
                    next_shard_getter = Shard.load_async(next(file_iter), world_size)
                except StopIteration:
                    next_shard_getter = None
                continue
            buf = torch.cat([tokens[i:j] for i, j in zip(start_idxs, end_idxs)])
            _inputs = buf[:-1]
            _targets = buf[1:]
            end_idxs[-1] -= 1
            cum_lengths = (end_idxs - start_idxs).cumsum(0)
        else:
            if pos + num_tokens + 1 >= len(tokens):
                tokens, pos = load_data_shard(next(file_iter)), 0
            pos_local = pos + rank * num_tokens_local
            buf = tokens[pos_local: pos_local + num_tokens_local + 1]
            _inputs = buf[:-1].view(num_tokens_local,)
            _targets = buf[1:].view(num_tokens_local,)
            cum_lengths = torch.nonzero(_inputs == BOS_ID)[:, 0]
            pos += num_tokens

        _cum_lengths = torch.full((max_num_docs,), num_tokens_local)
        _cum_lengths[0] = 0
        _cum_lengths[1:len(cum_lengths) + 1] = cum_lengths

        _inputs = _inputs.to(dtype=torch.int32)
        _targets = _targets.to(dtype=torch.int64)
        _cum_lengths = _cum_lengths.to(dtype=torch.int32)
        _bigram_inputs = get_bigram_hash(_inputs, bigram_vocab_size)

        new_params = yield (
            _inputs.to(device="cuda", non_blocking=True),
            _targets.to(device="cuda", non_blocking=True),
            _cum_lengths.to(device="cuda", non_blocking=True),
            _bigram_inputs.to(device="cuda", non_blocking=True),
            _bigram_inputs.numpy(),
        )

        if new_params is not None:
            new_num_tokens, new_max_seq_len, new_grad_accum_steps = new_params
            assert new_num_tokens % (world_size * new_grad_accum_steps) == 0, "Num tokens must be divisible by world size"
            num_tokens = new_num_tokens // new_grad_accum_steps
            max_seq_len = new_max_seq_len
