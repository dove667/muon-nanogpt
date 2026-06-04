
from dataclasses import dataclass


@dataclass(frozen=True)
class RunSpec:
    orth: str
    group: str
    name: str
    lr_mul: float
    seed: int = 0
    train_token_budget: int = 30_000_000
    eval_every_tokens: int = 5_000_000
    eval_tokens: int = 1_048_576
    ns_t: int | None = None
    fast_steps: int | None = None
    stable_steps: int | None = None
    pe_t: int | None = None
    pe_lower_bound: str | None = None

    def to_cli_args(self) -> list[str]:
        args = [
            "--orth", self.orth,
            "--group", self.group,
            "--name", self.name,
            "--lr-mul", str(self.lr_mul),
            "--seed", str(self.seed),
            "--train-token-budget", str(self.train_token_budget),
            "--eval-every-tokens", str(self.eval_every_tokens),
            "--eval-tokens", str(self.eval_tokens),
        ]
        if self.orth == "manual":
            args.extend([
                "--ns-t", str(self.ns_t),
                "--fast-steps", str(self.fast_steps),
                "--stable-steps", str(self.stable_steps),
            ])
        elif self.orth == "polar_express":
            args.extend([
                "--pe-t", str(self.pe_t),
                "--pe-lower-bound", str(self.pe_lower_bound),
            ])
        return args
