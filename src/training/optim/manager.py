import copy
import math

import torch

from model import ForwardScheduleConfig
from optim.core import NorMuonAndAdam
from config import TRAINING, MODEL, OPTIMIZER


def _compute_lr(step: int, total_steps: int, lr_mul: float = 1.0,
                warmup_frac: float | None = None, min_lr_frac: float | None = None) -> float:
    wf = TRAINING.warmup_frac if warmup_frac is None else warmup_frac
    mlf = TRAINING.min_lr_frac if min_lr_frac is None else min_lr_frac
    if total_steps <= 1:
        return lr_mul
    warmup_steps = max(1, round(total_steps * wf))
    if step < warmup_steps:
        return lr_mul * ((step + 1) / warmup_steps)
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(progress * math.pi))
    decay = mlf + (1.0 - mlf) * cosine
    return lr_mul * decay


def _build_param_table():
    raw = dict(OPTIMIZER.param_table.items())
    result = {}
    for label, cfg in raw.items():
        entry = {"optim": cfg["optim"]}
        if "adam_betas" in cfg:
            entry["adam_betas"] = list(cfg["adam_betas"])
        else:
            entry["adam_betas"] = None
        if "lr_mul" in cfg:
            entry["lr_mul"] = float(cfg["lr_mul"])
        if "wd_mul" in cfg:
            entry["wd_mul"] = float(cfg["wd_mul"])
        result[label] = entry
    return result


class TrainingManager:
    def __init__(self, model, *, device, total_steps, lr_mul, orth_mode, polar_express, grad_accum_steps):
        self.model = model
        self.device = device
        self.orth_mode = orth_mode
        self.total_steps = total_steps
        self.block_size = MODEL.block_size
        self.global_train_tokens = 0
        self.grad_accum_steps = grad_accum_steps
        self.train_loader_send_args = None

        self.param_table = _build_param_table()
        if self.orth_mode == "adamw":
            for label in ("qk_bank", "vo_bank", "mlp_bank"):
                self.param_table[label] = {"optim": "adam", "adam_betas": [0.9, 0.95]}

        self.work_order = [
            "scalars", "smear_gate", "skip_gate", "attn_gate_bank", "ve_gate_bank",
            "post_lambdas", "x0_lambdas", "bigram_lambdas", "resid_lambdas",
            "value_embeds", "bigram_embed", "lm_head", "embed",
            "qk_bank", "vo_bank", "mlp_bank",
        ]

        adam_cfg = OPTIMIZER.adam_defaults
        muon_cfg = OPTIMIZER.muon_defaults
        adam_defaults = dict(lr=adam_cfg.lr, eps=adam_cfg.eps, weight_decay=adam_cfg.weight_decay)
        normuon_defaults = dict(
            lr=muon_cfg.lr * lr_mul,
            momentum=muon_cfg.momentum,
            beta2=muon_cfg.beta2,
            weight_decay=muon_cfg.weight_decay,
        )

        self.optimizer = NorMuonAndAdam(
            model.named_parameters(),
            param_table=self.param_table,
            scatter_order=list(self.param_table),
            work_order=self.work_order,
            adam_defaults=adam_defaults,
            normuon_defaults=normuon_defaults,
            orthogonalize_fn=polar_express,
        )
        self.reset()

    def get_forward_args(self):
        return ForwardScheduleConfig(
            mtp_weights=self.mtp_weights,
            ws_short=self.ws_short * self.block_size,
            ws_long=self.ws_long * self.block_size,
            train_max_seq_len=self.train_max_seq_len,
        )

    def step_optimizers(self, step: int):
        step_lr = _compute_lr(step, self.total_steps)
        muon_momentum = OPTIMIZER.muon_defaults.momentum
        do_adam = self.orth_mode == "adamw" or step % OPTIMIZER.adam_step_interval == 1

        for _, param_cfg in self.optimizer.param_cfgs.items():
            param_cfg.lr = param_cfg.initial_lr * step_lr
            if param_cfg.optim == "normuon":
                param_cfg.momentum = muon_momentum

        self.optimizer.step(do_adam=do_adam)

    def reset(self, state=None):
        if state is not None:
            self.optimizer.load_state_dict(state)
        self.optimizer.reset()

        self.ws_short, self.ws_long = tuple(MODEL.window_sizes)
        self.batch_size = TRAINING.batch_tokens
        self.train_max_seq_len = TRAINING.seq_len
        self.mtp_weights = torch.tensor(list(MODEL.mtp_weights))
        self.model.yarn.reset()
        self.model.yarn_paired_head.reset()

    def get_state(self):
        return copy.deepcopy(self.optimizer.state_dict())
