import copy

import numpy as np
import torch

from model import ForwardScheduleConfig
from optim.core import NorMuonAndAdam
from schedule import get_muon_momentum
from optim.sparse_comms import sparse_comms_share_indexes, sparse_comms_start


class TrainingManager:
    def __init__(self, model, *, rank, world_size, grad_accum_steps, device, args, training_schedule, lr_mul, polar_express):
        self.model = model
        self.rank = rank
        self.world_size = world_size
        self.grad_accum_steps = grad_accum_steps
        self.device = device
        self.args = args
        self.training_schedule = training_schedule
        self.block_size = 128
        self.global_train_tokens = 0

        self.param_table = {
            "qk_bank": {"optim": "normuon", "comms": "sharded", "adam_betas": None},
            "vo_bank": {"optim": "normuon", "comms": "sharded", "adam_betas": None},
            "mlp_bank": {"optim": "normuon", "comms": "sharded", "adam_betas": None},
            "scalars": {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.99], "lr_mul": 5.0, "wd_mul": 0.0},
            "smear_gate": {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.99], "lr_mul": 0.01, "wd_mul": 0.0},
            "skip_gate": {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.99], "lr_mul": 0.05, "wd_mul": 0.0},
            "attn_gate_bank": {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.99]},
            "ve_gate_bank": {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.99]},
            "lm_head": {"optim": "adam", "comms": "sharded", "adam_betas": [0.5, 0.95], "wd_mul": 150.0},
            "bigram_embed": {"optim": "adam", "comms": "sharded_sparse", "adam_betas": [0.75, 0.95], "lr_mul": 75.0, "wd_mul": 5.0},
            "post_lambdas": {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.95], "lr_mul": 1.0, "wd_mul": 0.0},
            "x0_lambdas": {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.95], "lr_mul": 1.0, "wd_mul": 0.0},
            "bigram_lambdas": {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.95], "lr_mul": 1.0, "wd_mul": 0.0},
            "resid_lambdas": {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.95], "lr_mul": 5.0, "wd_mul": 0.0},
            "value_embeds": {"optim": "adam", "comms": "sharded", "adam_betas": [0.75, 0.95], "lr_mul": 75.0, "wd_mul": 5.0},
            "embed": {"optim": "adam", "comms": "sharded", "adam_betas": [0.5, 0.95], "wd_mul": 150.0},
        }
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
            world_size=world_size,
            grad_accum_steps=grad_accum_steps,
        )
        self.split_step = training_schedule.split_step
        self.reset()

    def _sparse_comms_active(self):
        return self.world_size == 8 and self.grad_accum_steps == 1

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
        do_adam = step % 2 == 1

        for _, param_cfg in self.optimizer.param_cfgs.items():
            param_cfg.lr = param_cfg.initial_lr * step_lr
            if param_cfg.optim == "normuon":
                param_cfg.momentum = muon_momentum

        self.optimizer.step(do_adam=do_adam)
        if step == self.split_step:
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
        if self._sparse_comms_active():
            self.row_update_mask = np.zeros(self.args.bigram_vocab_size, dtype=np.uint8)
            self.sparse_counts_state = None
            self.send_idxes_buffer = torch.empty(self.args.bigram_vocab_size, dtype=torch.int32, pin_memory=True)

    def get_state(self):
        return copy.deepcopy(self.optimizer.state_dict())

    def sparse_index_update(self, step, bigram_indexes):
        if not self._sparse_comms_active():
            return

        self.row_update_mask[bigram_indexes] = 1
        if step % 2 == 1:
            with torch.no_grad():
                bigram_idx_np = np.flatnonzero(self.row_update_mask).astype(np.int32)
                send_idxes, send_counts, recv_counts, recv_counts_fut = sparse_comms_start(
                    bigram_idx_np, self.args.bigram_vocab_size, self.rank, self.world_size, self.send_idxes_buffer,
                    device=self.device,
                )
                self.sparse_counts_state = (send_idxes, send_counts, recv_counts, recv_counts_fut)

    def sparse_index_share(self, step):
        if not self._sparse_comms_active() or step % 2 != 1:
            return

        send_idxes, send_counts, recv_counts, recv_counts_fut = self.sparse_counts_state
        self.sparse_counts_state = None

        recv_counts_fut.wait()
        recv_idxes, sparse_state, idxes_fut = sparse_comms_share_indexes(
            send_idxes, send_counts, recv_counts, device=self.device,
        )
        self.optimizer._reduce_futures[self.model.bigram_embed.weight] = [idxes_fut, recv_idxes]
        self.optimizer._sparse_async_data[self.model.bigram_embed.weight] = sparse_state
        self.row_update_mask.fill(0)
