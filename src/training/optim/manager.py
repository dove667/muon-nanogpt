import copy

import torch

from model import ForwardScheduleConfig
from optim.core import NorMuonAndAdam
from schedule import get_muon_momentum


class TrainingManager:
    def __init__(self, model, *, device, args, training_schedule, lr_mul, orth_mode, polar_express, grad_accum_steps):
        self.model = model
        self.device = device
        self.args = args
        self.training_schedule = training_schedule
        self.orth_mode = orth_mode
        self.block_size = 128
        self.global_train_tokens = 0
        self.grad_accum_steps = grad_accum_steps

        self.param_table = {
            "qk_bank": {"optim": "normuon", "adam_betas": None},
            "vo_bank": {"optim": "normuon", "adam_betas": None},
            "mlp_bank": {"optim": "normuon", "adam_betas": None},
            "scalars": {"optim": "adam", "adam_betas": [0.9, 0.99], "lr_mul": 5.0, "wd_mul": 0.0},
            "smear_gate": {"optim": "adam", "adam_betas": [0.9, 0.99], "lr_mul": 0.01, "wd_mul": 0.0},
            "skip_gate": {"optim": "adam", "adam_betas": [0.9, 0.99], "lr_mul": 0.05, "wd_mul": 0.0},
            "attn_gate_bank": {"optim": "adam", "adam_betas": [0.9, 0.99]},
            "ve_gate_bank": {"optim": "adam", "adam_betas": [0.9, 0.99]},
            "lm_head": {"optim": "adam", "adam_betas": [0.5, 0.95], "wd_mul": 150.0},
            "bigram_embed": {"optim": "adam", "adam_betas": [0.75, 0.95], "lr_mul": 75.0, "wd_mul": 5.0},
            "post_lambdas": {"optim": "adam", "adam_betas": [0.9, 0.95], "lr_mul": 1.0, "wd_mul": 0.0},
            "x0_lambdas": {"optim": "adam", "adam_betas": [0.9, 0.95], "lr_mul": 1.0, "wd_mul": 0.0},
            "bigram_lambdas": {"optim": "adam", "adam_betas": [0.9, 0.95], "lr_mul": 1.0, "wd_mul": 0.0},
            "resid_lambdas": {"optim": "adam", "adam_betas": [0.9, 0.95], "lr_mul": 5.0, "wd_mul": 0.0},
            "value_embeds": {"optim": "adam", "adam_betas": [0.75, 0.95], "lr_mul": 75.0, "wd_mul": 5.0},
            "embed": {"optim": "adam", "adam_betas": [0.5, 0.95], "wd_mul": 150.0},
        }
        if self.orth_mode == "adamw":
            for label in ("qk_bank", "vo_bank", "mlp_bank"):
                self.param_table[label] = {"optim": "adam", "adam_betas": [0.9, 0.95]}
        self.work_order = [
            "scalars", "smear_gate", "skip_gate", "attn_gate_bank", "ve_gate_bank",
            "post_lambdas", "x0_lambdas", "bigram_lambdas", "resid_lambdas",
            "value_embeds", "bigram_embed", "lm_head", "embed",
            "qk_bank", "vo_bank", "mlp_bank",
        ]

        adam_defaults = dict(lr=0.008, eps=1e-10, weight_decay=0.005)
        normuon_defaults = dict(lr=0.023 * lr_mul, momentum=0.95, beta2=0.9, weight_decay=1.2)

        self.optimizer = NorMuonAndAdam(
            model.named_parameters(),
            param_table=self.param_table,
            scatter_order=list(self.param_table),
            work_order=self.work_order,
            adam_defaults=adam_defaults,
            normuon_defaults=normuon_defaults,
            orthogonalize_fn=polar_express,
        )
        self.split_step = training_schedule.split_step
        self.reset()

    def apply_final_ws_ext(self):
        self.ws_long = self.training_schedule.ws_post_yarn_ext

    def get_forward_args(self):
        return ForwardScheduleConfig(
            mtp_weights=self.mtp_weights,
            ws_short=self.ws_short * self.block_size,
            ws_long=self.ws_long * self.block_size,
            train_max_seq_len=self.train_max_seq_len,
        )

    def advance_schedule(self, step: int):
        stage, _ = self.training_schedule.lookup(step)
        self.ws_short, new_ws_long = stage.window_sizes
        if new_ws_long != self.ws_long:
            self.model.yarn.apply(self.ws_long * self.block_size, new_ws_long * self.block_size)
            self.model.yarn_paired_head.apply(self.ws_long * self.block_size, new_ws_long * self.block_size)

        new_batch_size = stage.batch_size
        new_train_max_seq_len = stage.train_max_seq_len
        if new_batch_size != self.batch_size or new_train_max_seq_len != self.train_max_seq_len:
            self.train_loader_send_args = (new_batch_size, new_train_max_seq_len, self.grad_accum_steps)
            self.batch_size = new_batch_size
            self.train_max_seq_len = new_train_max_seq_len
        else:
            self.train_loader_send_args = None

        self.ws_long = new_ws_long
        self.mtp_weights = self.training_schedule.mtp_weights[step]

    def step_optimizers(self, step: int):
        step_lr = self.training_schedule.get_lr(step)
        muon_momentum = get_muon_momentum(step, self.training_schedule.total_steps)
        do_adam = self.orth_mode == "adamw" or step % 2 == 1

        for _, param_cfg in self.optimizer.param_cfgs.items():
            param_cfg.lr = param_cfg.initial_lr * step_lr
            if param_cfg.optim == "normuon":
                param_cfg.momentum = muon_momentum

        self.optimizer.step(do_adam=do_adam)
        if self.split_step is not None and step == self.split_step:
            self.optimizer.copy_lm_state_to_embed()

    def reset(self, state=None):
        if state is not None:
            self.optimizer.load_state_dict(state)
        self.optimizer.reset()

        stage, _ = self.training_schedule.lookup(0)
        self.ws_short, self.ws_long = stage.window_sizes
        self.batch_size = stage.batch_size
        self.train_max_seq_len = stage.train_max_seq_len
        self.model.yarn.reset()
        self.model.yarn_paired_head.reset()

    def get_state(self):
        return copy.deepcopy(self.optimizer.state_dict())

    def sparse_index_update(self, step, bigram_indexes):
        pass

    def sparse_index_share(self, step):
        pass
