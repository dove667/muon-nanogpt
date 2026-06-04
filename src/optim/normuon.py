from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(slots=True)
class ParamConfig:
    label: str
    optim: str
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
    ):
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
        self._init_state()

    def _build_param_cfg(self, param: nn.Parameter, label: str) -> None:
        table_entry = self.param_table[label]
        optim = table_entry["optim"]
        adam_betas = table_entry.get("adam_betas")
        lr_mul = table_entry.get("lr_mul", 1.0)
        wd_mul = table_entry.get("wd_mul", 1.0)

        if optim == "adam":
            param_cfg = ParamConfig(
                label=label,
                optim=optim,
                adam_betas=tuple(adam_betas) if adam_betas else None,
                lr_mul=lr_mul,
                wd_mul=wd_mul,
                lr=self.adam_defaults["lr"],
                initial_lr=self.adam_defaults["lr"],
                weight_decay=self.adam_defaults["weight_decay"],
                eps=self.adam_defaults["eps"],
            )
        elif optim == "normuon":
            reshape = getattr(param, "reshape", None)
            if not isinstance(reshape, tuple):
                reshape = (1, *param.shape) if param.ndim == 2 else tuple(param.shape)
            chunk_size = reshape[0]
            chunk_shape = (chunk_size, *reshape[1:])
            shape_mult = max(1.0, chunk_shape[-2] / chunk_shape[-1]) ** 0.5 if len(chunk_shape) >= 2 else 1.0
            param_cfg = ParamConfig(
                label=label,
                optim=optim,
                adam_betas=tuple(adam_betas) if adam_betas else None,
                lr_mul=shape_mult * lr_mul,
                wd_mul=wd_mul,
                lr=self.normuon_defaults["lr"],
                initial_lr=self.normuon_defaults["lr"],
                weight_decay=self.normuon_defaults["weight_decay"],
                reshape=reshape,
                chunk_size=chunk_size,
                momentum=self.normuon_defaults["momentum"],
                beta2=self.normuon_defaults["beta2"],
            )
        else:
            raise ValueError(f"Unknown optim type: {optim}")

        self.param_cfgs[param] = param_cfg

    def _init_state(self) -> None:
        for param, param_cfg in self.param_cfgs.items():
            if param_cfg.optim == "adam":
                exp_avg = torch.zeros_like(param, dtype=torch.float32, device=param.device)
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

    def reset(self) -> None:
        for param, param_cfg in self.param_cfgs.items():
            if param_cfg.optim != "normuon":
                continue
            state = self.param_states[param]
            state["momentum_buffer"].zero_()
            state["second_momentum_buffer"].zero_()

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
        for label in self.scatter_order:
            param = self._param_by_label[label]
            param_cfg = self.param_cfgs[param]
            if param_cfg.optim == "adam" and not do_adam:
                continue
            if param.grad is None:
                continue

            grad = param.grad
            if param_cfg.optim == "normuon":
                grad = grad.view(param_cfg.reshape)

            if param_cfg.optim == "adam":
                self._adam_update(param, grad, param_cfg)
            else:
                self._normuon_update(param, grad, param_cfg)

        for param, param_cfg in self.param_cfgs.items():
            if param_cfg.optim == "adam" and not do_adam:
                continue
            param.grad = None

    def _adam_update(self, param: nn.Parameter, grad_chunk: Tensor, param_cfg: ParamConfig) -> None:
        beta1, beta2 = param_cfg.adam_betas
        lr = param_cfg.lr * param_cfg.lr_mul

        state = self.param_states[param]
        state["step"] += 1
        step = state["step"]

        bias1 = 1 - beta1**step
        bias2 = 1 - beta2**step
        step_size = lr / bias1
        bias2_sqrt = bias2**0.5

        exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
        exp_avg.mul_(beta1).add_(grad_chunk, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad_chunk, grad_chunk, value=1 - beta2)
        denom = exp_avg_sq.sqrt().div_(bias2_sqrt).add_(param_cfg.eps)
        param.mul_(1 - lr * param_cfg.weight_decay * param_cfg.wd_mul)
        param.addcdiv_(exp_avg, denom, value=-step_size)

    def _normuon_update(self, param: nn.Parameter, grad_chunk: Tensor, param_cfg: ParamConfig) -> None:
        state = self.param_states[param]
        grad_chunk = grad_chunk.float()

        eff_lr = param_cfg.lr_mul * param_cfg.lr
        eff_wd = param_cfg.wd_mul * param_cfg.weight_decay * param_cfg.lr
        momentum_t = torch.tensor(param_cfg.momentum, dtype=torch.float32, device="cpu")
        value_chunk = self.orthogonalize_fn(
            grad_chunk,
            state["momentum_buffer"],
            momentum_t,
            split_baddbmm=grad_chunk.shape[-2] > 1024,
        )

        red_dim = -1 if grad_chunk.shape[-2] >= grad_chunk.shape[-1] else -2
        value_chunk = self._apply_normuon_variance_reduction(
            value_chunk,
            state["second_momentum_buffer"],
            param_cfg.beta2,
            red_dim,
        )
        _cautious_wd_update(param.data.view(param_cfg.reshape), value_chunk, eff_wd, eff_lr)

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
