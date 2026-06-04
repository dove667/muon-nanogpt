from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import Tensor, nn

from optim.sparse_comms import sparse_comms_merge_gradients, sparse_comms_share_gradients


def _transpose_copy(src: Tensor, dst: Tensor) -> None:
    dst.copy_(src.T)


def _transpose_add(src: Tensor, dst: Tensor) -> None:
    dst.add_(src.T)


@dataclass(slots=True)
class ParamConfig:
    label: str
    optim: str
    comms: str
    adam_betas: tuple[float, float] | None
    lr_mul: float
    wd_mul: float
    lr: float
    initial_lr: float
    weight_decay: float
    eps: float | None = None
    reshape: tuple | None = None
    chunk_size: int | None = None
    momentum: float | None = None
    beta2: float | None = None
    per_matrix_lr_mul: list[float] | None = None


class NorMuonAndAdam:
    def __init__(
        self,
        named_params,
        param_table: dict,
        scatter_order: list,
        work_order: list,
        adam_defaults: dict,
        normuon_defaults: dict,
        orthogonalize_fn: callable,
        world_size: int,
        grad_accum_steps: int,
    ):
        self.world_size = world_size
        self.grad_accum_steps = grad_accum_steps
        self.orthogonalize_fn = orthogonalize_fn
        self.adam_defaults = adam_defaults
        self.normuon_defaults = normuon_defaults
        self.param_table = param_table
        self.scatter_order = scatter_order
        self.work_order = work_order

        self.param_cfgs: dict[nn.Parameter, ParamConfig] = {}
        self.param_states: dict[nn.Parameter, dict] = {}
        self._param_by_label: dict[str, nn.Parameter] = {}
        for _, param in named_params:
            label = getattr(param, "label", None)
            assert label is not None and label in param_table
            assert label not in self._param_by_label
            self._param_by_label[label] = param
            self._build_param_cfg(param, label)

        present = self._param_by_label.keys()
        assert set(scatter_order) == present and set(work_order) == present

        if self.world_size == 1:
            for param_cfg in self.param_cfgs.values():
                param_cfg.comms = "none"

        self._init_state()

        self._reduce_futures: dict[nn.Parameter, tuple] = {}
        self._sparse_async_data: dict[nn.Parameter, list] = {}

        self.split_embed = False
        self._lm_head_param = self._param_by_label.get("lm_head")
        self._embed_param = self._param_by_label.get("embed")

    def _build_param_cfg(self, param: nn.Parameter, label: str):
        table_entry = self.param_table[label]
        optim = table_entry["optim"]
        comms = table_entry["comms"]
        if comms == "sharded_sparse" and not (self.world_size == 8 and self.grad_accum_steps == 1):
            comms = "sharded"

        adam_betas = table_entry.get("adam_betas")
        lr_mul = table_entry.get("lr_mul", 1.0)
        wd_mul = table_entry.get("wd_mul", 1.0)

        if optim == "adam":
            chunk_size = param.shape[0] // self.world_size if comms.startswith("sharded") else None
            param_cfg = ParamConfig(
                label=label,
                optim=optim,
                comms=comms,
                adam_betas=tuple(adam_betas) if adam_betas else None,
                lr_mul=lr_mul,
                wd_mul=wd_mul,
                lr=self.adam_defaults["lr"],
                initial_lr=self.adam_defaults["lr"],
                weight_decay=self.adam_defaults["weight_decay"],
                eps=self.adam_defaults["eps"],
                chunk_size=chunk_size,
            )
        elif optim == "normuon":
            reshape = getattr(param, "reshape", None)
            if reshape is None:
                raise ValueError(f"NorMuon param {label} must have .reshape attribute")
            if reshape[0] % self.world_size != 0:
                raise ValueError(f"reshape[0]={reshape[0]} must be divisible by world_size")

            chunk_size = reshape[0] // self.world_size
            chunk_shape = (chunk_size, *reshape[1:])
            shape_mult = max(1.0, chunk_shape[-2] / chunk_shape[-1]) ** 0.5 if len(chunk_shape) >= 2 else 1.0
            lr_mul = shape_mult * lr_mul

            per_matrix_lr_mul = None
            if label == "mlp_bank":
                rank = dist.get_rank() if dist.is_initialized() else 0
                start_idx = rank * chunk_size
                per_matrix_lr_mul = []
                for idx in range(chunk_size):
                    global_idx = start_idx + idx
                    per_matrix_lr_mul.append(2.0 if global_idx % 2 == 1 else 1.0)

            param_cfg = ParamConfig(
                label=label,
                optim=optim,
                comms=comms,
                adam_betas=tuple(adam_betas) if adam_betas else None,
                lr_mul=lr_mul,
                wd_mul=wd_mul,
                lr=self.normuon_defaults["lr"],
                initial_lr=self.normuon_defaults["lr"],
                weight_decay=self.normuon_defaults["weight_decay"],
                reshape=reshape,
                chunk_size=chunk_size,
                momentum=self.normuon_defaults["momentum"],
                beta2=self.normuon_defaults["beta2"],
                per_matrix_lr_mul=per_matrix_lr_mul,
            )
        else:
            raise ValueError(f"Unknown optim type: {optim}")

        self.param_cfgs[param] = param_cfg

    def _init_state(self):
        for param, param_cfg in self.param_cfgs.items():
            if param_cfg.optim == "adam":
                chunk = param[: param_cfg.chunk_size] if param_cfg.comms.startswith("sharded") else param
                exp_avg = torch.zeros_like(chunk, dtype=torch.float32, device=param.device)
                self.param_states[param] = dict(step=0, exp_avg=exp_avg, exp_avg_sq=torch.zeros_like(exp_avg))
                continue

            chunk_shape = (param_cfg.chunk_size, *param_cfg.reshape[1:])
            momentum_buffer = torch.zeros(chunk_shape, dtype=torch.float32, device=param.device)
            if chunk_shape[-2] >= chunk_shape[-1]:
                second_mom_shape = (*chunk_shape[:-1], 1)
            else:
                second_mom_shape = (*chunk_shape[:-2], 1, chunk_shape[-1])
            second_momentum_buffer = torch.zeros(second_mom_shape, dtype=torch.float32, device=param.device)
            self.param_states[param] = dict(
                momentum_buffer=momentum_buffer,
                second_momentum_buffer=second_momentum_buffer,
            )

    def _launch_reduce(self, param: nn.Parameter, grad: Tensor):
        param_cfg = self.param_cfgs[param]

        if param_cfg.comms == "none":
            if param_cfg.optim == "normuon":
                grad = grad.view(param_cfg.reshape)
            self._reduce_futures[param] = (None, grad)
            return

        if param_cfg.comms == "replicated":
            future = dist.all_reduce(grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
            self._reduce_futures[param] = (future, grad)
            return

        if param_cfg.comms == "sharded":
            if param_cfg.optim == "normuon":
                grad_reshaped = grad.view(param_cfg.reshape)
                grad_chunk = torch.empty(
                    (param_cfg.chunk_size, *grad_reshaped.shape[1:]),
                    dtype=grad.dtype,
                    device=grad.device,
                )
                future = dist.reduce_scatter_tensor(
                    grad_chunk,
                    grad_reshaped.contiguous(),
                    op=dist.ReduceOp.AVG,
                    async_op=True,
                ).get_future()
                self._reduce_futures[param] = (future, grad_chunk)
            else:
                grad_chunk = torch.empty_like(grad[: param_cfg.chunk_size])
                future = dist.reduce_scatter_tensor(
                    grad_chunk,
                    grad,
                    op=dist.ReduceOp.AVG,
                    async_op=True,
                ).get_future()
                self._reduce_futures[param] = (future, grad_chunk)
            return

        sparse_state = self._sparse_async_data[param]
        recv_vals, val_fut = sparse_comms_share_gradients(
            grad,
            sparse_state["send_idxes"],
            sparse_state["send_counts"],
            sparse_state["recv_counts"],
        )
        self._reduce_futures[param].extend((val_fut, recv_vals))

    def _launch_gather(self, param: nn.Parameter, param_slice: Tensor) -> "torch.futures.Future":
        param_cfg = self.param_cfgs[param]
        if param_cfg.optim == "normuon":
            full_param = param.data.view(param_cfg.reshape)
            assert full_param.is_contiguous()
            return dist.all_gather_into_tensor(
                full_param,
                param_slice.contiguous(),
                async_op=True,
            ).get_future()
        return dist.all_gather_into_tensor(param, param_slice.contiguous(), async_op=True).get_future()

    def reset(self):
        self.split_embed = False
        for param, param_cfg in self.param_cfgs.items():
            if param_cfg.optim != "normuon":
                continue
            state = self.param_states[param]
            state["momentum_buffer"].zero_()
            state["second_momentum_buffer"].zero_()

    def copy_lm_state_to_embed(self):
        lm_head = self._lm_head_param
        embed = self._embed_param
        lm_state = self.param_states[lm_head]
        embed_state = self.param_states[embed]
        lm_cfg = self.param_cfgs[lm_head]
        embed_cfg = self.param_cfgs[embed]

        embed_state["step"] = lm_state["step"]

        if self.world_size > 1:
            rank = dist.get_rank()
            embed_chunk_size = embed_cfg.chunk_size
            for key in ["exp_avg", "exp_avg_sq"]:
                lm_chunk = lm_state[key]
                full_lm = torch.empty(
                    lm_head.shape[0],
                    lm_head.shape[1],
                    dtype=lm_chunk.dtype,
                    device=lm_chunk.device,
                )
                dist.all_gather_into_tensor(full_lm, lm_chunk.contiguous())
                embed_state[key].copy_(
                    full_lm.T[rank * embed_chunk_size : (rank + 1) * embed_chunk_size]
                )
        else:
            for key in ["exp_avg", "exp_avg_sq"]:
                embed_state[key].copy_(lm_state[key].T)

        self.split_embed = True

    def state_dict(self):
        return {
            "param_states": {id(param): state for param, state in self.param_states.items()},
            "param_cfgs": {id(param): state for param, state in self.param_cfgs.items()},
        }

    def load_state_dict(self, state_dict):
        id_to_param = {id(param): param for param in self.param_cfgs}
        for param_id, saved_state in state_dict["param_states"].items():
            if param_id not in id_to_param:
                continue
            param = id_to_param[param_id]
            current_state = self.param_states[param]
            for key, value in saved_state.items():
                if isinstance(value, torch.Tensor) and key in current_state:
                    current_state[key] = value.to(
                        dtype=current_state[key].dtype,
                        device=current_state[key].device,
                    )
                else:
                    current_state[key] = value

    @torch.no_grad()
    def step(self, do_adam: bool = True):
        rank = dist.get_rank() if dist.is_initialized() else 0
        lm_param, embed_param = self._lm_head_param, self._embed_param

        for label in self.scatter_order:
            param = self._param_by_label[label]
            param_cfg = self.param_cfgs[param]
            if param_cfg.optim == "adam" and not do_adam:
                continue
            if param.grad is None:
                continue
            if label == "lm_head" and do_adam and not self.split_embed:
                if embed_param is not None and embed_param.grad is not None:
                    _transpose_add(embed_param.grad, param.grad)
            if label == "embed" and not self.split_embed:
                continue
            self._launch_reduce(param, param.grad)

        gather_futures = []
        lm_head_gather_future = None

        for label in self.work_order:
            param = self._param_by_label[label]
            if param not in self._reduce_futures:
                continue
            param_cfg = self.param_cfgs[param]
            if param_cfg.optim == "adam" and not do_adam:
                continue

            if param_cfg.comms != "sharded_sparse":
                future, grad_chunk = self._reduce_futures[param]
                if future is not None:
                    future.wait()
            else:
                idxes_fut, recv_idxes, recv_fut, recv_vals = self._reduce_futures[param]
                idxes_fut.wait()
                recv_fut.wait()
                grad_chunk = sparse_comms_merge_gradients(param.grad, recv_idxes, recv_vals, rank, self.world_size)

            if param_cfg.optim == "adam":
                param_slice = self._adam_update(param, grad_chunk, param_cfg, rank)
            else:
                param_slice = self._normuon_update(param, grad_chunk, param_cfg, rank)

            if param_cfg.comms.startswith("sharded") and self.world_size > 1:
                gather_future = self._launch_gather(param, param_slice)
                if label == "lm_head":
                    lm_head_gather_future = gather_future
                else:
                    gather_futures.append(gather_future)

        if lm_head_gather_future is not None:
            lm_head_gather_future.wait()

        if do_adam and not self.split_embed and embed_param is not None and lm_param is not None:
            _transpose_copy(lm_param.data, embed_param.data)

        for future in gather_futures:
            future.wait()

        self._reduce_futures.clear()
        self._sparse_async_data.clear()

        for param, param_cfg in self.param_cfgs.items():
            if param_cfg.optim == "adam" and not do_adam:
                continue
            param.grad = None

    def _adam_update(self, param: nn.Parameter, grad_chunk: Tensor, param_cfg: ParamConfig, rank: int) -> Tensor:
        beta1, beta2 = param_cfg.adam_betas
        lr = param_cfg.lr * param_cfg.lr_mul
        param_slice = (
            param[rank * param_cfg.chunk_size : (rank + 1) * param_cfg.chunk_size]
            if param_cfg.comms.startswith("sharded")
            else param
        )

        state = self.param_states[param]
        state["step"] += 1
        step = state["step"]

        bias1 = 1 - beta1**step
        bias2 = 1 - beta2**step
        step_size = lr * (bias2**0.5 / bias1)
        eff_wd = lr * lr * param_cfg.weight_decay * param_cfg.wd_mul

        exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
        exp_avg.mul_(beta1).add_(grad_chunk, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad_chunk, grad_chunk, value=1 - beta2)
        update = exp_avg.div(exp_avg_sq.sqrt().add_(param_cfg.eps)).mul_(step_size)
        mask = (update * param_slice) > 0
        update.addcmul_(param_slice, mask, value=eff_wd)
        param_slice.add_(other=update, alpha=-1.0)
        return param_slice

    def _normuon_update(self, param: nn.Parameter, grad_chunk: Tensor, param_cfg: ParamConfig, rank: int) -> Tensor:
        state = self.param_states[param]
        grad_chunk = grad_chunk.float()

        momentum_val = param_cfg.momentum
        eff_lr = param_cfg.lr_mul * param_cfg.lr
        eff_wd = param_cfg.wd_mul * param_cfg.weight_decay * param_cfg.lr

        is_large_matrix = grad_chunk.shape[-2] > 1024
        momentum_t = torch.tensor(momentum_val, dtype=torch.float32, device="cpu")
        value_chunk = self.orthogonalize_fn(
            grad_chunk,
            state["momentum_buffer"],
            momentum_t,
            split_baddbmm=is_large_matrix,
        )

        red_dim = -1 if grad_chunk.shape[-2] >= grad_chunk.shape[-1] else -2
        value_chunk = self._apply_normuon_variance_reduction(
            value_chunk,
            state["second_momentum_buffer"],
            param_cfg.beta2,
            red_dim,
        )

        param_view = param.data.view(param_cfg.reshape)
        param_slice = param_view[rank * param_cfg.chunk_size : (rank + 1) * param_cfg.chunk_size]

        if param_cfg.per_matrix_lr_mul is not None:
            for mat_idx in range(param_cfg.chunk_size):
                individual_lr = eff_lr * param_cfg.per_matrix_lr_mul[mat_idx]
                _cautious_wd_update(param_slice[mat_idx], value_chunk[mat_idx], eff_wd, individual_lr)
        else:
            _cautious_wd_update(param_slice, value_chunk, eff_wd, eff_lr)

        return param_slice

    @staticmethod
    def _apply_normuon_variance_reduction(value_chunk, second_momentum_buffer, beta2, red_dim):
        value_mean = value_chunk.float().square().mean(dim=red_dim, keepdim=True)
        red_dim_size = value_chunk.size(red_dim)
        value_norm_sq = value_mean.sum(dim=(-2, -1), keepdim=True).mul_(red_dim_size)
        value_norm = value_norm_sq.sqrt_()
        second_momentum_buffer.lerp_(value_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
        step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt_()
        scaled_sq_sum = (value_mean * red_dim_size) * step_size.float().square()
        value_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt_()
        final_scale = step_size * (value_norm / value_norm_new.clamp_min_(1e-10))
        return value_chunk.mul_(final_scale.type_as(value_chunk))


def _cautious_wd_update(param, grad, wd, lr):
    grad_f = grad.float()
    param_f = param.float()
    mask = (grad_f * param_f) >= 0
    update = grad_f * lr + (param_f * mask.to(param_f.dtype) * wd * lr)
    param.copy_((param_f - update).to(dtype=param.dtype))
