
from src.plan.models import RunSpec
from src.plan.selection import (
    best_pe_t5_lower_bound,
    best_record,
    completed_records,
    record_key,
    run_completed_any_group,
    top_pe_lr_expand_specs,
)


DEFAULT_BUDGET = 30_000_000
DEFAULT_EVAL_EVERY = 5_000_000
DEFAULT_EVAL_TOKENS = 1_048_576
DEFAULT_MAIN_BUDGET = 100_000_000
DEFAULT_MAIN_EVAL_EVERY = 10_000_000
DEFAULT_MAIN_EVAL_TOKENS = 2_097_152
DEFAULT_MAIN_TOP_N = 8
DEFAULT_FINAL_BUDGET = 300_000_000
DEFAULT_FINAL_EVAL_EVERY = 20_000_000
DEFAULT_FINAL_EVAL_TOKENS = 5_242_880


def emit_vanilla(lr, *, group="vanilla_muon", seed=0, budget=DEFAULT_BUDGET, eval_every=DEFAULT_EVAL_EVERY, eval_tokens=DEFAULT_EVAL_TOKENS, name=None):
    return RunSpec("vanilla", group, name or f"old_fast5_lr{lr}_seed{seed}", lr, seed=seed, train_token_budget=budget, eval_every_tokens=eval_every, eval_tokens=eval_tokens)


def emit_manual(ns_t, fast_steps, stable_steps, lr, *, group, seed=0, budget=DEFAULT_BUDGET, eval_every=DEFAULT_EVAL_EVERY, eval_tokens=DEFAULT_EVAL_TOKENS, name=None):
    return RunSpec("manual", group, name or f"p2_T{ns_t}_f{fast_steps}_s{stable_steps}_lr{lr}_seed{seed}", lr, seed=seed, train_token_budget=budget, eval_every_tokens=eval_every, eval_tokens=eval_tokens, ns_t=ns_t, fast_steps=fast_steps, stable_steps=stable_steps)


def emit_pe(pe_t, lower_bound, lr, *, group, seed=0, budget=DEFAULT_BUDGET, eval_every=DEFAULT_EVAL_EVERY, eval_tokens=DEFAULT_EVAL_TOKENS, name=None):
    return RunSpec("polar_express", group, name or f"pe_T{pe_t}_l{lower_bound}_lr{lr}_seed{seed}", lr, seed=seed, train_token_budget=budget, eval_every_tokens=eval_every, eval_tokens=eval_tokens, pe_t=pe_t, pe_lower_bound=lower_bound)


def iter_vanilla(*, budget=DEFAULT_BUDGET, eval_every=DEFAULT_EVAL_EVERY, eval_tokens=DEFAULT_EVAL_TOKENS):
    for lr in (1.0, 0.5, 2.0):
        yield emit_vanilla(lr, budget=budget, eval_every=eval_every, eval_tokens=eval_tokens)


def iter_p2_t5(*, budget=DEFAULT_BUDGET, eval_every=DEFAULT_EVAL_EVERY, eval_tokens=DEFAULT_EVAL_TOKENS):
    for fast_steps in range(6):
        stable_steps = 5 - fast_steps
        for lr in (0.5, 1.0, 2.0):
            yield emit_manual(5, fast_steps, stable_steps, lr, group="p2_T5_lrgrid", budget=budget, eval_every=eval_every, eval_tokens=eval_tokens)


def iter_pe_init(*, budget=DEFAULT_BUDGET, eval_every=DEFAULT_EVAL_EVERY, eval_tokens=DEFAULT_EVAL_TOKENS):
    for lower_bound in ("1e-2", "1e-3", "1e-4"):
        for lr in (0.5, 1.0, 2.0):
            yield emit_pe(5, lower_bound, lr, group="pe_init", budget=budget, eval_every=eval_every, eval_tokens=eval_tokens)


def iter_p2_t10(*, budget=DEFAULT_BUDGET, eval_every=DEFAULT_EVAL_EVERY, eval_tokens=DEFAULT_EVAL_TOKENS):
    for fast_steps in range(11):
        yield emit_manual(10, fast_steps, 10 - fast_steps, 1.0, group="p2_T10", budget=budget, eval_every=eval_every, eval_tokens=eval_tokens)


def iter_p2_t78(*, budget=DEFAULT_BUDGET, eval_every=DEFAULT_EVAL_EVERY, eval_tokens=DEFAULT_EVAL_TOKENS):
    for ns_t in (7, 8):
        for fast_steps in range(ns_t + 1):
            yield emit_manual(ns_t, fast_steps, ns_t - fast_steps, 1.0, group="p2_T78", budget=budget, eval_every=eval_every, eval_tokens=eval_tokens)


def iter_p2_t69(*, budget=DEFAULT_BUDGET, eval_every=DEFAULT_EVAL_EVERY, eval_tokens=DEFAULT_EVAL_TOKENS):
    for ns_t in (6, 9):
        for fast_steps in range(ns_t + 1):
            yield emit_manual(ns_t, fast_steps, ns_t - fast_steps, 1.0, group="p2_T69", budget=budget, eval_every=eval_every, eval_tokens=eval_tokens)


def iter_pe_lower_expand(*, budget=DEFAULT_BUDGET, eval_every=DEFAULT_EVAL_EVERY, eval_tokens=DEFAULT_EVAL_TOKENS):
    for lower_bound in ("3e-3", "3e-4", "3e-2", "3e-5"):
        yield emit_pe(5, lower_bound, 1.0, group="pe_expand", budget=budget, eval_every=eval_every, eval_tokens=eval_tokens)


def iter_pe_iter_expand(*, budget=DEFAULT_BUDGET, eval_every=DEFAULT_EVAL_EVERY, eval_tokens=DEFAULT_EVAL_TOKENS):
    best_lower = best_pe_t5_lower_bound()
    print("=" * 80)
    print(f"PE iteration expansion best_l={best_lower}")
    print("=" * 80)
    seen = set()
    for lower_bound in (best_lower, "1e-3", "1e-4"):
        if lower_bound in seen:
            continue
        seen.add(lower_bound)
        for pe_t in (6, 7, 8, 9, 10):
            yield emit_pe(pe_t, lower_bound, 1.0, group="pe_expand", budget=budget, eval_every=eval_every, eval_tokens=eval_tokens)


def iter_pe_lr_expand(*, budget=DEFAULT_BUDGET, eval_every=DEFAULT_EVAL_EVERY, eval_tokens=DEFAULT_EVAL_TOKENS):
    for pe_t, lower_bound in top_pe_lr_expand_specs():
        for lr in (0.5, 2.0):
            name = f"pe_T{pe_t}_l{lower_bound}_lr{lr}_seed0"
            if run_completed_any_group(name):
                print("=" * 80)
                print(f"skip completed in another group: {name}")
                print("=" * 80)
                continue
            yield emit_pe(pe_t, lower_bound, lr, group="pe_expand", name=name, budget=budget, eval_every=eval_every, eval_tokens=eval_tokens)


def iter_pe_expand(*, budget=DEFAULT_BUDGET, eval_every=DEFAULT_EVAL_EVERY, eval_tokens=DEFAULT_EVAL_TOKENS):
    yield from iter_pe_lower_expand(budget=budget, eval_every=eval_every, eval_tokens=eval_tokens)
    yield from iter_pe_iter_expand(budget=budget, eval_every=eval_every, eval_tokens=eval_tokens)
    yield from iter_pe_lr_expand(budget=budget, eval_every=eval_every, eval_tokens=eval_tokens)


def spec_from_record(record, *, group, prefix, budget, eval_every, eval_tokens, seed):
    config = record["config"]
    orth = config.get("orthogonalizer_type")
    schedule = config.get("orth_schedule_name")
    lr = float(config.get("lr_mul", 1.0))
    budget_label = f"{budget // 1_000_000}M"
    name = f"{prefix}_{schedule}_lr{lr}_{budget_label}_seed{seed}"
    if orth == "vanilla":
        return emit_vanilla(lr, group=group, seed=seed, budget=budget, eval_every=eval_every, eval_tokens=eval_tokens, name=name)
    if orth == "manual":
        return emit_manual(int(config.get("T_ns")), int(config.get("fast_steps")), int(config.get("stable_steps")), lr, group=group, seed=seed, budget=budget, eval_every=eval_every, eval_tokens=eval_tokens, name=name)
    if orth == "polar_express":
        return emit_pe(int(config.get("pe_T")), str(config.get("pe_lower_bound")), lr, group=group, seed=seed, budget=budget, eval_every=eval_every, eval_tokens=eval_tokens, name=name)
    raise RuntimeError(f"Unknown orthogonalizer {orth!r} in record {record['name']}")


def iter_main(*, budget=DEFAULT_MAIN_BUDGET, eval_every=DEFAULT_MAIN_EVAL_EVERY, eval_tokens=DEFAULT_MAIN_EVAL_TOKENS, top_n=DEFAULT_MAIN_TOP_N):
    records = completed_records(excluded_groups={"calibration", "main", "final"})
    if not records:
        raise SystemExit("No completed grid records found for main selection")
    selected = []
    for record in [
        best_record(records, lambda r: r["config"].get("orthogonalizer_type") == "vanilla" and abs(float(r["config"].get("lr_mul", 1.0)) - 1.0) < 1e-12),
        best_record(records, lambda r: r["config"].get("orthogonalizer_type") == "manual" and int(r["config"].get("T_ns", -1)) == 5),
        best_record(records, lambda r: r["config"].get("orthogonalizer_type") == "manual" and int(r["config"].get("T_ns", -1)) >= 6),
        best_record(records, lambda r: r["config"].get("orthogonalizer_type") == "polar_express"),
    ]:
        if record is not None:
            selected.append(record)
    selected.extend(sorted(records, key=lambda record: record["val"]))
    seen = set()
    count = 0
    for record in selected:
        key = record_key(record)
        if key in seen:
            continue
        seen.add(key)
        print(f"main selected from {record['name']}: {record['config'].get('orth_schedule_name')} lr={float(record['config'].get('lr_mul', 1.0))}")
        yield spec_from_record(record, group="main", prefix="main", budget=budget, eval_every=eval_every, eval_tokens=eval_tokens, seed=0)
        count += 1
        if count >= top_n:
            break


def iter_final(*, budget=DEFAULT_FINAL_BUDGET, eval_every=DEFAULT_FINAL_EVAL_EVERY, eval_tokens=DEFAULT_FINAL_EVAL_TOKENS):
    records = completed_records(included_groups={"main"})
    if not records:
        records = completed_records(included_groups={"vanilla_muon", "p2_T5_lrgrid", "pe_init", "p2_T10", "p2_T78", "p2_T69", "pe_expand"})
    if not records:
        raise SystemExit("No completed records found for final selection")
    selected = [
        best_record(records, lambda r: r["config"].get("orthogonalizer_type") == "manual"),
        best_record(records, lambda r: r["config"].get("orthogonalizer_type") == "polar_express"),
        best_record(records, lambda r: r["config"].get("orthogonalizer_type") == "vanilla" and abs(float(r["config"].get("lr_mul", 1.0)) - 1.0) < 1e-12),
    ]
    seen = set()
    for record in selected:
        if record is None:
            continue
        config = record["config"]
        key = (config.get("orthogonalizer_type"), config.get("orth_schedule_name"), float(config.get("lr_mul", 1.0)))
        if key in seen:
            continue
        seen.add(key)
        for seed in (0, 1, 2):
            print(f"final selected from {record['name']}: {config.get('orth_schedule_name')} lr={float(config.get('lr_mul', 1.0))} seed={seed}")
            yield spec_from_record(record, group="final", prefix="final", budget=budget, eval_every=eval_every, eval_tokens=eval_tokens, seed=seed)


def iter_minimal(*, budget=DEFAULT_BUDGET, eval_every=DEFAULT_EVAL_EVERY, eval_tokens=DEFAULT_EVAL_TOKENS):
    yield from iter_vanilla(budget=budget, eval_every=eval_every, eval_tokens=eval_tokens)
    yield from iter_p2_t5(budget=budget, eval_every=eval_every, eval_tokens=eval_tokens)
    yield from iter_pe_init(budget=budget, eval_every=eval_every, eval_tokens=eval_tokens)
    yield from iter_p2_t10(budget=budget, eval_every=eval_every, eval_tokens=eval_tokens)


def iter_core75(*, budget=DEFAULT_BUDGET, eval_every=DEFAULT_EVAL_EVERY, eval_tokens=DEFAULT_EVAL_TOKENS):
    yield from iter_vanilla(budget=budget, eval_every=eval_every, eval_tokens=eval_tokens)
    yield from iter_p2_t5(budget=budget, eval_every=eval_every, eval_tokens=eval_tokens)
    yield from iter_pe_init(budget=budget, eval_every=eval_every, eval_tokens=eval_tokens)
    yield from iter_p2_t10(budget=budget, eval_every=eval_every, eval_tokens=eval_tokens)
    yield from iter_p2_t78(budget=budget, eval_every=eval_every, eval_tokens=eval_tokens)
    yield from iter_p2_t69(budget=budget, eval_every=eval_every, eval_tokens=eval_tokens)


def iter_main_final(*, budget=DEFAULT_BUDGET, eval_every=DEFAULT_EVAL_EVERY, eval_tokens=DEFAULT_EVAL_TOKENS, main_budget=DEFAULT_MAIN_BUDGET, main_eval_every=DEFAULT_MAIN_EVAL_EVERY, main_eval_tokens=DEFAULT_MAIN_EVAL_TOKENS, main_top_n=DEFAULT_MAIN_TOP_N, final_budget=DEFAULT_FINAL_BUDGET, final_eval_every=DEFAULT_FINAL_EVAL_EVERY, final_eval_tokens=DEFAULT_FINAL_EVAL_TOKENS):
    del budget, eval_every, eval_tokens
    yield from iter_main(budget=main_budget, eval_every=main_eval_every, eval_tokens=main_eval_tokens, top_n=main_top_n)
    yield from iter_final(budget=final_budget, eval_every=final_eval_every, eval_tokens=final_eval_tokens)
