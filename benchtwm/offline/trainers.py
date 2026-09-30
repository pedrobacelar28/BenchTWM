"""Offline trainers for fixed transition datasets."""

from __future__ import annotations

import copy
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from benchtwm.utils.progress import progress_write, tqdm
from benchtwm.utils.queue_progress import update_progress

from benchtwm.algorithms.cem_world_models import batch_loss as _cem_batch_loss
from benchtwm.algorithms.cem_world_models import build_model as _build_cem_model
from benchtwm.algorithms.cem_world_models import make_policy as _make_cem_policy
from benchtwm.algorithms.cem_world_models import rollout_validation_metrics as _cem_rollout_validation_metrics
from benchtwm.algorithms.bc import BCPolicy
from benchtwm.algorithms.cql import CQLPolicy
from benchtwm.algorithms.dqn import DQNPolicy
from benchtwm.algorithms.dreamer import DreamerPolicy
from benchtwm.algorithms.iql import IQLPolicy
from benchtwm.evaluation import evaluate_policy
from benchtwm.models.mlp import build_actor_critic, build_q_network
from benchtwm.models.dreamer import DreamerAgent, DreamerConfig, DreamerOptimizer
from benchtwm.models.offline_rl import build_cql_agent, build_iql_agent
from benchtwm.offline.dataset import OfflineTransitionDataset
from benchtwm.offline.dreamer_dataset import DreamerSequenceStream
from benchtwm.utils.metrics import summarize_training
from benchtwm.utils.methods import method_id
from benchtwm.utils.seeding import set_seed
from benchtwm.utils.torch_utils import get_torch_device


@dataclass
class OfflineProgress:
    step: int
    sampled_transitions: int
    log_due: bool
    eval_due: bool
    bar: Any | None = None


@dataclass
class OfflineSchedule:
    total_steps: int
    batch_size: int
    total_sampled_transitions_estimate: int


@dataclass
class EarlyStoppingState:
    enabled: bool
    patience: int
    min_delta: float
    best_loss: float = float("inf")
    best_step: int = -1
    best_sampled_transitions: int = -1
    best_checkpoint: dict[str, Any] | None = None
    checks_without_improvement: int = 0
    num_checks: int = 0
    early_stopped: bool = False
    early_stop_step: int = -1
    early_stop_sampled_transitions: int = -1
    early_stop_reason: str = "disabled"


def _validate_step_config(args) -> None:
    required = ["offline_train_steps", "offline_batch_size", "offline_log_interval", "offline_eval_interval"]
    missing = [key for key in required if not hasattr(args, key)]
    if missing:
        raise ValueError(f"Offline config is missing required step-based fields: {missing}")
    if int(args.offline_train_steps) <= 0:
        raise ValueError("offline_train_steps must be > 0")
    if int(args.offline_batch_size) <= 0:
        raise ValueError("offline_batch_size must be > 0")


def _batch_size(args) -> int:
    return int(args.offline_batch_size)


def _eval_episodes(args) -> int:
    return int(getattr(args, "offline_eval_episodes", getattr(args, "eval_episodes", 10)))


def _show_progress(args) -> bool:
    return bool(getattr(args, "show_progress", True))


def _use_tqdm_bars(args) -> bool:
    return _show_progress(args) and sys.stderr.isatty()


def _console_style(args) -> str:
    return str(getattr(args, "offline_console_style", "compact")).lower()


def _format_metric(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{value:.4f}"


def _short_metric_name(key: str) -> str:
    names = {
        "train_loss": "train_loss",
        "validation_loss": "val_loss",
        "validation_transitions": "val_trans",
        "mean_action_stability_l1": "stability_l1",
        "mean_abrupt_jump_rate": "abrupt",
        "mean_return": "return",
        "mean_success": "success",
        "mean_survival": "survival",
        "mean_length": "length",
        "validation_dynamics_loss": "val_dyn",
        "validation_reward_loss": "val_rew",
        "validation_continue_loss": "val_cont",
        "validation_prediction_error": "val_pred_err",
        "validation_mask_loss": "val_mask",
        "validation_policy_loss": "val_policy",
        "validation_value_loss": "val_value",
        "validation_bc_loss": "val_bc",
        "validation_entropy": "val_entropy",
        "validation_q_loss": "val_q",
        "validation_cql_loss": "val_cql",
        "validation_actor_loss": "val_actor",
    }
    return names.get(key, key)


def _write_metrics(args, tag: str, progress: OfflineProgress | None, rows: dict[str, float | int]) -> None:
    if not _show_progress(args):
        return
    if _console_style(args) == "verbose":
        suffix = "" if progress is None else f" @ step={progress.step}"
        progress_write(f"{tag}{suffix}:")
        for key, value in rows.items():
            if isinstance(value, float):
                progress_write(f"  {key}: {value:.6f}")
            else:
                progress_write(f"  {key}: {value}")
        return
    prefix = f"[{tag.lower()}]"
    if progress is not None:
        prefix = f"{prefix} step={progress.step:>6d} sampled={progress.sampled_transitions:>9d}"
    metrics = "  ".join(f"{_short_metric_name(key)}={_format_metric(value)}" for key, value in rows.items())
    progress_write(f"{prefix}  {metrics}")


def _set_train_postfix(progress: OfflineProgress, train_loss: float, extra: dict[str, float] | None = None) -> None:
    if progress.bar is None:
        return
    postfix: dict[str, float | int | str] = {
        "loss": f"{train_loss:.4f}",
        "sampled": progress.sampled_transitions,
    }
    if extra:
        for key, value in extra.items():
            postfix[key] = f"{float(value):.4f}"
    progress.bar.set_postfix(postfix)


def _schedule(args, dataset: OfflineTransitionDataset) -> OfflineSchedule:
    _validate_step_config(args)
    total_steps = int(args.offline_train_steps)
    batch_size = _batch_size(args)
    total_sampled_transitions = total_steps * batch_size
    return OfflineSchedule(
        total_steps=total_steps,
        batch_size=batch_size,
        total_sampled_transitions_estimate=total_sampled_transitions,
    )


def _log_interval(args) -> int:
    return int(getattr(args, "offline_log_interval", 0))


def _eval_interval(args) -> int:
    return int(getattr(args, "offline_eval_interval", 0))


def _eval_during_training(args) -> bool:
    return bool(getattr(args, "offline_eval_during_training", True))


def _make_early_stopping_state(args) -> EarlyStoppingState:
    patience = int(getattr(args, "early_stopping_patience", 20))
    enabled = patience > 0
    return EarlyStoppingState(
        enabled=enabled,
        patience=max(0, patience),
        min_delta=float(getattr(args, "early_stopping_min_delta", 0.0)),
        early_stop_reason="completed_budget" if enabled else "disabled",
    )


def _update_early_stopping(
    state: EarlyStoppingState,
    *,
    validation_loss: float,
    progress: OfflineProgress,
    checkpoint: dict[str, Any],
    checkpoint_is_snapshot: bool = False,
) -> tuple[bool, bool]:
    state.num_checks += 1
    improved = validation_loss < (state.best_loss - state.min_delta)
    if improved or state.best_checkpoint is None:
        state.best_loss = float(validation_loss)
        state.best_step = int(progress.step)
        state.best_sampled_transitions = int(progress.sampled_transitions)
        state.best_checkpoint = checkpoint if checkpoint_is_snapshot else copy.deepcopy(checkpoint)
        state.checks_without_improvement = 0
        return True, False
    state.checks_without_improvement += 1
    if state.enabled and state.checks_without_improvement >= state.patience:
        state.early_stopped = True
        state.early_stop_step = int(progress.step)
        state.early_stop_sampled_transitions = int(progress.sampled_transitions)
        state.early_stop_reason = "patience_exhausted"
        return False, True
    return False, False


def _report_early_stopping(
    args,
    state: EarlyStoppingState,
    *,
    progress: OfflineProgress,
    validation_loss: float,
    improved: bool,
    stop_due: bool,
) -> None:
    """Emit the criterion and patience state alongside each validation check."""

    if not _show_progress(args) or not state.enabled:
        return
    decision = "new best" if improved else f"no improvement ({state.checks_without_improvement}/{state.patience})"
    if stop_due:
        decision = "EARLY STOP: patience exhausted"
    progress_write(
        "[early-stop] "
        f"step={progress.step} monitor=validation_loss(min) current={validation_loss:.6f} "
        f"best={state.best_loss:.6f} min_delta={state.min_delta:g} {decision}"
    )


def _finalize_early_stopping(state: EarlyStoppingState, *, progress: OfflineProgress | None, schedule: OfflineSchedule) -> dict[str, Any]:
    if state.best_checkpoint is None:
        raise RuntimeError("validation-best checkpoint was never set")
    if not state.early_stopped:
        if progress is not None:
            state.early_stop_step = int(progress.step)
            state.early_stop_sampled_transitions = int(progress.sampled_transitions)
        else:
            state.early_stop_step = int(schedule.total_steps)
            state.early_stop_sampled_transitions = int(schedule.total_sampled_transitions_estimate)
        if state.enabled:
            state.early_stop_reason = "completed_budget"
    return {
        "best_validation_loss": float(state.best_loss),
        "best_validation_step": int(state.best_step),
        "best_validation_sampled_transitions": int(state.best_sampled_transitions),
        "early_stopped": bool(state.early_stopped),
        "early_stop_step": int(state.early_stop_step),
        "early_stop_sampled_transitions": int(state.early_stop_sampled_transitions),
        "early_stop_reason": str(state.early_stop_reason),
        "num_validation_checks": int(state.num_checks),
        "num_checks_without_improvement": int(state.checks_without_improvement),
    }


def _load_datasets(args, obs_backend: str):
    _validate_step_config(args)
    seed = int(args.seed)
    train_dataset = OfflineTransitionDataset(args.dataset_path, obs_backend=obs_backend, seed=seed)
    val_dataset = OfflineTransitionDataset(args.val_dataset_path, obs_backend=obs_backend, seed=seed + 100000)
    return train_dataset, val_dataset


def _eval_stats(policy, config: dict, progress: OfflineProgress, args) -> dict:
    checkpoint_name = f"offline_step_{progress.step}"
    result = evaluate_policy(
        policy,
        config=config,
        num_episodes=_eval_episodes(args),
        eval_seed_base=int(args.eval_seed_base),
        checkpoint_name=checkpoint_name,
        export_csv=False,
        show_progress=_show_progress(args),
        progress_desc=f"eval step={progress.step}",
    )
    episodes = result["episodes"]
    summary_metrics = {
        key: float(value)
        for key, value in result["summary"].items()
        if key.startswith("mean_") and isinstance(value, (int, float, np.number))
    }
    stats = {
        "step": int(progress.step),
        "sampled_transitions": int(progress.sampled_transitions),
        **summary_metrics,
        "action_stability_l1": float(episodes["action_stability_l1"].mean()) if not episodes.empty else 0.0,
        "abrupt_jump_rate": float(episodes["abrupt_jump_rate"].mean()) if not episodes.empty else 0.0,
    }
    _write_metrics(
        args,
        "eval",
        progress,
        {
            "episodes": _eval_episodes(args),
            **summary_metrics,
            "mean_action_stability_l1": stats["action_stability_l1"],
            "mean_abrupt_jump_rate": stats["abrupt_jump_rate"],
        },
    )
    return stats


def _steps_to_threshold(eval_history: list[dict], threshold: float) -> int:
    for row in eval_history:
        if float(row["mean_return"]) >= threshold:
            return int(row["step"])
    return -1


def _metrics_from_history(
    train_history: list[dict],
    eval_history: list[dict],
    args,
    config: dict,
    train_dataset: OfflineTransitionDataset,
    val_dataset: OfflineTransitionDataset,
    schedule: OfflineSchedule,
    training_metadata: dict[str, Any] | None = None,
) -> dict:
    returns = np.asarray([row["mean_return"] for row in eval_history], dtype=np.float32)
    success = np.asarray(
        [row.get("mean_success", row.get("mean_survival", 0.0)) for row in eval_history],
        dtype=np.float32,
    )
    lengths = np.asarray([row["mean_length"] for row in eval_history], dtype=np.float32)
    update_steps = np.asarray([row["step"] for row in eval_history], dtype=np.int32)
    sampled_transitions = np.asarray([row.get("sampled_transitions", 0) for row in eval_history], dtype=np.int32)
    stability = np.asarray([row["action_stability_l1"] for row in eval_history], dtype=np.float32)
    abrupt = np.asarray([row["abrupt_jump_rate"] for row in eval_history], dtype=np.float32)
    summary = summarize_training(
        returns=returns,
        episode_lengths=lengths,
        total_steps=update_steps,
        action_stability_l1=stability,
        abrupt_jump_rate=abrupt,
        threshold=float(args.success_threshold),
        success_rates=success,
    )
    summary.pop("total_env_steps", None)
    summary.pop("episodes_to_threshold", None)
    actual_total_steps = int(train_history[-1]["step"]) if train_history else int(schedule.total_steps)
    actual_total_sampled_transitions = (
        int(train_history[-1]["sampled_transitions"])
        if train_history
        else int(actual_total_steps * schedule.batch_size)
    )
    effective_epochs = float(actual_total_sampled_transitions / max(1, len(train_dataset)))
    summary.update(
        {
            "offline_train_steps": int(schedule.total_steps),
            "offline_batch_size": int(schedule.batch_size),
            "total_gradient_steps": actual_total_steps,
            "total_sampled_transitions": actual_total_sampled_transitions,
            "total_sampled_transitions_estimate": int(schedule.total_sampled_transitions_estimate),
            "effective_epochs": effective_epochs,
            "best_eval_return": float(np.max(returns)) if returns.size else 0.0,
            "steps_to_threshold": _steps_to_threshold(eval_history, float(args.success_threshold)),
            "updates_to_threshold": _steps_to_threshold(eval_history, float(args.success_threshold)),
            "train_dataset_transitions": int(len(train_dataset)),
            "val_dataset_transitions": int(len(val_dataset)),
            "train_dataset_episodes": int(train_dataset.num_episodes),
            "val_dataset_episodes": int(val_dataset.num_episodes),
        }
    )
    if training_metadata:
        summary.update(training_metadata)
    extra_metrics: dict[str, np.ndarray] = {}
    if train_history:
        keys = sorted(key for key in train_history[0] if key not in {"step", "sampled_transitions"})
        for key in keys:
            extra_metrics[key] = np.asarray([row.get(key, np.nan) for row in train_history], dtype=np.float32)
    return {
        "returns": returns,
        "discounted_returns": returns.copy(),
        "num_steps": lengths.astype(np.int32),
        "episode_length": lengths.astype(np.int32),
        "total_steps": update_steps,
        "offline_updates": update_steps,
        "offline_sampled_transitions": sampled_transitions,
        "success_rate": success,
        "survival_rate": success,
        "action_stability_l1": stability,
        "abrupt_jump_rate": abrupt,
        "extra_metrics": extra_metrics,
        "offline_training_history": {
            "step": np.asarray([row["step"] for row in train_history], dtype=np.int32),
            "sampled_transitions": np.asarray([row.get("sampled_transitions", 0) for row in train_history], dtype=np.int32),
            **{
                key: np.asarray([row.get(key, np.nan) for row in train_history], dtype=np.float32)
                for key in (sorted(k for k in train_history[0] if k not in {"step", "sampled_transitions"}) if train_history else [])
            },
        },
        "offline_eval_history": {
            "step": update_steps,
            "sampled_transitions": sampled_transitions,
            **{
                f"eval_{key}": np.asarray(
                    [row.get(key, np.nan) for row in eval_history],
                    dtype=np.float32,
                )
                for key in sorted(
                    {
                        key
                        for row in eval_history
                        for key in row
                        if key not in {"step", "sampled_transitions"}
                    }
                )
            },
        },
        "summary": summary,
    }


def _training_batches(args, dataset: OfflineTransitionDataset, device: str | torch.device, desc: str):
    schedule = _schedule(args, dataset)
    disable = not _use_tqdm_bars(args)
    progress_bar = tqdm(
        range(1, schedule.total_steps + 1),
        desc=desc,
        unit="step",
        dynamic_ncols=True,
        disable=disable,
    )
    try:
        for step in progress_bar:
            batch = dataset.sample(_batch_size(args), device=device)
            sampled_transitions = step * _batch_size(args)
            is_last = step == schedule.total_steps
            log_due = is_last or (_log_interval(args) > 0 and step % _log_interval(args) == 0)
            eval_due = _eval_during_training(args) and (
                is_last or (_eval_interval(args) > 0 and step % _eval_interval(args) == 0)
            )
            if step == 1 or log_due:
                update_progress(
                    status="running",
                    phase="optimization",
                    method_id=desc,
                    step=step,
                    total_steps=schedule.total_steps,
                    sampled_transitions=sampled_transitions,
                )
            yield OfflineProgress(step, sampled_transitions, log_due, eval_due, progress_bar), batch, schedule
    finally:
        # Trainers may break early after a validation check. Close the shared
        # Rich task so final evaluation and OPE can render their own progress.
        progress_bar.close()


def _validation_loss(
    val_dataset: OfflineTransitionDataset,
    args,
    device: str | torch.device,
    loss_fn,
    *,
    progress: OfflineProgress | None = None,
    train_loss: float | None = None,
):
    total_loss = 0.0
    total_count = 0
    summed_info: dict[str, float] = {}
    with torch.no_grad():
        for batch in val_dataset.iter_transition_batches(_batch_size(args), device=device, shuffle=False):
            loss, info = loss_fn(batch)
            batch_count = int(batch.action.shape[0])
            total_loss += float(loss.item()) * batch_count
            total_count += batch_count
            for key, value in info.items():
                summed_info[key] = summed_info.get(key, 0.0) + float(value) * batch_count
    if total_count <= 0:
        raise RuntimeError(f"Validation dataset {val_dataset.path} produced no batches")
    loss_value = total_loss / total_count
    averaged = {key: value / total_count for key, value in summed_info.items()}
    update_progress(
        status="running",
        phase="optimization",
        step=progress.step if progress is not None else None,
        total_steps=getattr(args, "offline_train_steps", None),
        validation_loss=loss_value,
    )
    rows: dict[str, float | int] = {}
    if train_loss is not None:
        rows["train_loss"] = float(train_loss)
    rows["validation_loss"] = loss_value
    rows["validation_transitions"] = int(total_count)
    for key, value in averaged.items():
        rows[f"validation_{key}"] = float(value)
    _write_metrics(args, "val", progress, rows)
    return loss_value, averaged


def _final_result(metrics: dict, last_checkpoint: dict, best_checkpoint: dict) -> dict:
    return {
        "metrics": metrics,
        "checkpoint_last": last_checkpoint,
        "checkpoint_best": best_checkpoint,
    }


def _dqn_loss(network, target_network, batch, args) -> torch.Tensor:
    with torch.no_grad():
        target_next = target_network(batch.next_obs, batch.next_action_mask).max(dim=1).values
        td_target = batch.reward + float(args.gamma) * target_next * (1.0 - batch.done)
    old_val = network(batch.obs).gather(1, batch.action.unsqueeze(1)).squeeze(1)
    return F.mse_loss(old_val, td_target)


def train_dqn(config: dict, output_dir):
    args = SimpleNamespace(**config)
    set_seed(args.seed)
    device = get_torch_device()
    train_dataset, val_dataset = _load_datasets(args, args.obs_backend)
    q_network = build_q_network(
        config,
        obs_dim=train_dataset.obs_dim,
        num_actions=train_dataset.num_actions,
    ).to(device)
    target_network = build_q_network(
        config,
        obs_dim=train_dataset.obs_dim,
        num_actions=train_dataset.num_actions,
    ).to(device)
    target_network.load_state_dict(q_network.state_dict())
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    train_history = []
    eval_history = []
    early_stopping = _make_early_stopping_state(args)

    last_schedule = _schedule(args, train_dataset)
    last_progress: OfflineProgress | None = None
    for progress, batch, schedule in _training_batches(args, train_dataset, device, method_id(config)):
        last_schedule = schedule
        last_progress = progress
        loss = _dqn_loss(q_network, target_network, batch, args)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _set_train_postfix(progress, float(loss.item()))
        stop_due = False
        if progress.step % int(args.target_network_frequency) == 0:
            for target_param, source_param in zip(target_network.parameters(), q_network.parameters()):
                target_param.data.copy_(float(args.tau) * source_param.data + (1.0 - float(args.tau)) * target_param.data)
        if progress.log_due:
            val_loss, _ = _validation_loss(
                val_dataset,
                args,
                device,
                lambda val_batch: (_dqn_loss(q_network, target_network, val_batch, args), {}),
                progress=progress,
                train_loss=float(loss.item()),
            )
            train_history.append(
                {
                    "step": progress.step,
                    "sampled_transitions": progress.sampled_transitions,
                    "loss": float(loss.item()),
                    "validation_loss": float(val_loss),
                    "td_loss": float(loss.item()),
                    "validation_td_loss": float(val_loss),
                }
            )
            _set_train_postfix(progress, float(loss.item()), {"val_loss": val_loss})
            improved, stop_due = _update_early_stopping(
                early_stopping,
                validation_loss=float(val_loss),
                progress=progress,
                checkpoint={"model_state_dict": q_network.state_dict()},
            )
            _report_early_stopping(
                args,
                early_stopping,
                progress=progress,
                validation_loss=float(val_loss),
                improved=improved,
                stop_due=stop_due,
            )
        if progress.eval_due:
            stats = _eval_stats(DQNPolicy(q_network, temperature=float(getattr(args, "policy_temperature", 1.0))), config, progress, args)
            eval_history.append(stats)
            _set_train_postfix(progress, float(loss.item()), {"eval_return": stats["mean_return"]})
        if stop_due:
            break

    training_metadata = _finalize_early_stopping(early_stopping, progress=last_progress, schedule=last_schedule)
    metrics = _metrics_from_history(
        train_history,
        eval_history,
        args,
        config,
        train_dataset,
        val_dataset,
        last_schedule,
        training_metadata=training_metadata,
    )
    return _final_result(metrics, {"model_state_dict": q_network.state_dict()}, early_stopping.best_checkpoint)


def _bc_losses(agent, batch, args) -> tuple[torch.Tensor, dict]:
    logprob, entropy, _ = agent.evaluate_actions(batch.obs, batch.action, batch.action_mask)
    loss = -logprob.mean()
    return loss, {"bc_loss": float(loss.item()), "entropy": float(entropy.mean().item())}


def _iql_losses(agent, batch, args) -> tuple[torch.Tensor, dict]:
    q1_values = agent.q1(batch.obs, batch.action_mask)
    q2_values = agent.q2(batch.obs, batch.action_mask)
    q1_action = q1_values.gather(1, batch.action.unsqueeze(1)).squeeze(1)
    q2_action = q2_values.gather(1, batch.action.unsqueeze(1)).squeeze(1)
    q_min_action = torch.minimum(q1_action, q2_action)

    with torch.no_grad():
        next_value = agent.value(batch.next_obs).view(-1)
        td_target = batch.reward + float(args.gamma) * next_value * (1.0 - batch.done)
    q_loss = F.mse_loss(q1_action, td_target) + F.mse_loss(q2_action, td_target)

    value = agent.value(batch.obs).view(-1)
    value_error = q_min_action.detach() - value
    expectile = float(getattr(args, "iql_expectile_tau", getattr(args, "expectile_tau", 0.7)))
    value_loss = torch.where(value_error < 0.0, expectile, 1.0 - expectile) * value_error.pow(2)
    value_loss = value_loss.mean()

    advantage = (q_min_action.detach() - value.detach())
    beta = float(getattr(args, "iql_temperature_beta", getattr(args, "temperature_beta", 3.0)))
    clip = float(getattr(args, "iql_advantage_clip", getattr(args, "advantage_clip", 100.0)))
    weights = torch.exp(beta * advantage).clamp(max=clip)
    logprob, entropy, _ = agent.actor.evaluate_actions(batch.obs, batch.action, batch.action_mask)
    actor_loss = -(weights * logprob).mean()

    loss = (
        float(getattr(args, "iql_q_loss_weight", getattr(args, "q_loss_weight", 1.0))) * q_loss
        + float(getattr(args, "iql_value_loss_weight", getattr(args, "value_loss_weight", 1.0))) * value_loss
        + float(getattr(args, "iql_actor_loss_weight", getattr(args, "actor_loss_weight", 1.0))) * actor_loss
    )
    return loss, {
        "q_loss": float(q_loss.item()),
        "value_loss": float(value_loss.item()),
        "actor_loss": float(actor_loss.item()),
        "entropy": float(entropy.mean().item()),
        "advantage_weight": float(weights.mean().item()),
    }


def _cql_losses(agent, batch, args) -> tuple[torch.Tensor, dict]:
    q1_values = agent.q1(batch.obs, batch.action_mask)
    q2_values = agent.q2(batch.obs, batch.action_mask)
    q1_action = q1_values.gather(1, batch.action.unsqueeze(1)).squeeze(1)
    q2_action = q2_values.gather(1, batch.action.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_q = torch.minimum(
            agent.q1(batch.next_obs, batch.next_action_mask),
            agent.q2(batch.next_obs, batch.next_action_mask),
        ).max(dim=1).values
        td_target = batch.reward + float(args.gamma) * next_q * (1.0 - batch.done)
    td_loss = F.mse_loss(q1_action, td_target) + F.mse_loss(q2_action, td_target)
    # The Q networks return -1e9 for inadmissible actions, so logsumexp is
    # exactly over the admissible action set rather than a post-hoc penalty.
    conservative = (
        torch.logsumexp(q1_values, dim=1).mean() - q1_action.mean()
        + torch.logsumexp(q2_values, dim=1).mean() - q2_action.mean()
    ) * 0.5
    alpha = float(getattr(args, "cql_alpha", 1.0))
    loss = td_loss + alpha * conservative
    return loss, {
        "q_loss": float(td_loss.item()),
        "cql_loss": float(conservative.item()),
        "conservative_loss": float((alpha * conservative).item()),
    }


def _train_offline_policy(config: dict, output_dir, *, agent, loss_fn, make_policy, tag: str):
    """Shared step-budget loop for BC/IQL/CQL.

    These methods deliberately use the same validation/checkpoint contract as
    the existing offline DQN trainer.  Simulator evaluation is controlled by
    the recipe; the v0 recipes disable intermediate evaluation and the runner
    performs the fixed final evaluation.
    """
    args = SimpleNamespace(**config)
    set_seed(args.seed)
    device = get_torch_device()
    agent = agent.to(device)
    train_dataset, val_dataset = _load_datasets(args, args.obs_backend)
    optimizer = optim.Adam(agent.parameters(), lr=float(args.learning_rate))
    train_history = []
    eval_history = []
    early_stopping = _make_early_stopping_state(args)
    last_schedule = _schedule(args, train_dataset)
    last_progress: OfflineProgress | None = None

    for progress, batch, schedule in _training_batches(args, train_dataset, device, tag):
        last_schedule = schedule
        last_progress = progress
        loss, info = loss_fn(agent, batch, args)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), float(getattr(args, "max_grad_norm", 10.0)))
        optimizer.step()
        _set_train_postfix(progress, float(loss.item()), info)
        stop_due = False
        if progress.log_due:
            val_loss, val_info = _validation_loss(
                val_dataset,
                args,
                device,
                lambda val_batch: loss_fn(agent, val_batch, args),
                progress=progress,
                train_loss=float(loss.item()),
            )
            train_history.append(
                {
                    "step": progress.step,
                    "sampled_transitions": progress.sampled_transitions,
                    "loss": float(loss.item()),
                    "validation_loss": float(val_loss),
                    **info,
                    **{f"validation_{key}": value for key, value in val_info.items()},
                }
            )
            _set_train_postfix(progress, float(loss.item()), {"val_loss": val_loss})
            improved, stop_due = _update_early_stopping(
                early_stopping,
                validation_loss=float(val_loss),
                progress=progress,
                checkpoint={"model_state_dict": agent.state_dict()},
            )
            _report_early_stopping(
                args,
                early_stopping,
                progress=progress,
                validation_loss=float(val_loss),
                improved=improved,
                stop_due=stop_due,
            )
        if progress.eval_due:
            stats = _eval_stats(make_policy(agent), config, progress, args)
            eval_history.append(stats)
            _set_train_postfix(progress, float(loss.item()), {"eval_return": stats["mean_return"]})
        if stop_due:
            break

    training_metadata = _finalize_early_stopping(early_stopping, progress=last_progress, schedule=last_schedule)
    metrics = _metrics_from_history(
        train_history,
        eval_history,
        args,
        config,
        train_dataset,
        val_dataset,
        last_schedule,
        training_metadata=training_metadata,
    )
    return _final_result(metrics, {"model_state_dict": agent.state_dict()}, early_stopping.best_checkpoint)


def train_bc(config: dict, output_dir):
    args = SimpleNamespace(**config)
    # Dataset dimensions are read once before constructing the actor so both
    # vector and retained discrete_id recipes use the same builder.
    train_dataset, _ = _load_datasets(args, args.obs_backend)
    agent = build_actor_critic(
        config,
        obs_dim=train_dataset.obs_dim,
        num_actions=train_dataset.num_actions,
    )
    return _train_offline_policy(config, output_dir, agent=agent, loss_fn=_bc_losses, make_policy=lambda a: BCPolicy(a), tag=method_id(config))


def train_iql(config: dict, output_dir):
    args = SimpleNamespace(**config)
    train_dataset, _ = _load_datasets(args, args.obs_backend)
    agent = build_iql_agent(
        config,
        obs_dim=train_dataset.obs_dim,
        num_actions=train_dataset.num_actions,
    )
    return _train_offline_policy(config, output_dir, agent=agent, loss_fn=_iql_losses, make_policy=lambda a: IQLPolicy(a), tag=method_id(config))


def train_cql(config: dict, output_dir):
    args = SimpleNamespace(**config)
    train_dataset, _ = _load_datasets(args, args.obs_backend)
    agent = build_cql_agent(
        config,
        obs_dim=train_dataset.obs_dim,
        num_actions=train_dataset.num_actions,
    )
    return _train_offline_policy(config, output_dir, agent=agent, loss_fn=_cql_losses, make_policy=lambda a: CQLPolicy(a), tag=method_id(config))


def _train_cem_model(config: dict, output_dir, *, model_family: str):
    args = SimpleNamespace(**config)
    set_seed(args.seed)
    device = get_torch_device()
    train_dataset, val_dataset = _load_datasets(args, args.obs_backend)
    model = _build_cem_model(
        model_family,
        obs_dim=train_dataset.obs_dim,
        num_actions=train_dataset.num_actions,
        hidden_dim=args.hidden_dim,
        action_embed_dim=args.action_embed_dim,
        latent_dim=getattr(args, "latent_dim", None),
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    train_history = []
    eval_history = []
    early_stopping = _make_early_stopping_state(args)
    latent_model_target_decay = float(args.target_encoder_decay) if model_family == "latent_model" else None

    last_schedule = _schedule(args, train_dataset)
    last_progress: OfflineProgress | None = None
    for progress, batch, schedule in _training_batches(args, train_dataset, device, method_id(config)):
        last_schedule = schedule
        last_progress = progress
        loss, info = _cem_batch_loss(model, batch, args, model_family=model_family)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if model_family == "latent_model":
            model.update_target_encoder(latent_model_target_decay)
        postfix = {
            "dyn": info["dynamics_loss"],
            "rew": info["reward_loss"],
            "cont": info["continue_loss"],
        }
        if "reconstruction_loss" in info:
            postfix["recon"] = info["reconstruction_loss"]
        _set_train_postfix(progress, float(loss.item()), postfix)
        stop_due = False
        if progress.log_due:
            val_loss, val_info = _validation_loss(
                val_dataset,
                args,
                device,
                lambda val_batch: _cem_batch_loss(
                    model,
                    val_batch,
                    args,
                    model_family=model_family,
                ),
                progress=progress,
                train_loss=float(loss.item()),
            )
            rollout_info = _cem_rollout_validation_metrics(
                model,
                val_dataset,
                args,
                model_family=model_family,
            )
            train_history.append(
                {
                    "step": progress.step,
                    "sampled_transitions": progress.sampled_transitions,
                    "loss": float(loss.item()),
                    "validation_loss": float(val_loss),
                    **info,
                    **{f"validation_{key}": value for key, value in val_info.items()},
                    **rollout_info,
                }
            )
            _set_train_postfix(progress, float(loss.item()), {"val_loss": val_loss})
            improved, stop_due = _update_early_stopping(
                early_stopping,
                validation_loss=float(val_loss),
                progress=progress,
                checkpoint={"model_state_dict": model.state_dict()},
            )
            _report_early_stopping(
                args,
                early_stopping,
                progress=progress,
                validation_loss=float(val_loss),
                improved=improved,
                stop_due=stop_due,
            )
        if progress.eval_due:
            stats = _eval_stats(_make_cem_policy(model, config), config, progress, args)
            eval_history.append(stats)
            _set_train_postfix(progress, float(loss.item()), {"eval_return": stats["mean_return"]})
        if stop_due:
            break
    training_metadata = _finalize_early_stopping(early_stopping, progress=last_progress, schedule=last_schedule)
    metrics = _metrics_from_history(
        train_history,
        eval_history,
        args,
        config,
        train_dataset,
        val_dataset,
        last_schedule,
        training_metadata=training_metadata,
    )
    return _final_result(metrics, {"model_state_dict": model.state_dict()}, early_stopping.best_checkpoint)


def train_raw_model(config: dict, output_dir):
    return _train_cem_model(config, output_dir, model_family="raw_model")


def train_latent_model(config: dict, output_dir):
    return _train_cem_model(config, output_dir, model_family="latent_model")


def train_reconstructive_model(config: dict, output_dir):
    return _train_cem_model(config, output_dir, model_family="reconstructive_model")


def _dreamer_checkpoint(agent: DreamerAgent) -> dict:
    return {
        "checkpoint_format": "benchtwm-dreamer-v1",
        "dreamer_config": agent.config.asdict(),
        # A 200M Dreamer checkpoint must not duplicate the full model on GPU
        # during validation-selected checkpointing.
        "model_state_dict": {
            key: value.detach().cpu().clone()
            for key, value in agent.state_dict().items()
        },
    }


def _dreamer_validation(
    agent: DreamerAgent,
    dataset: OfflineTransitionDataset,
    args,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    stream = DreamerSequenceStream(
        dataset,
        batch_size=int(args.offline_batch_size),
        sequence_length=int(getattr(args, "dreamer_batch_length", 64)),
        seed=int(args.seed) + 200000,
        shuffle=False,
        cycle=False,
    )
    total_loss = 0.0
    total_records = 0
    totals: dict[str, float] = {}
    carry = None
    previous_action = None
    was_training = agent.training
    agent.eval()
    devices = [device.index] if device.type == "cuda" and device.index is not None else []
    with torch.random.fork_rng(devices=devices), torch.no_grad():
        torch.manual_seed(int(args.seed) + 300000)
        while True:
            batch = stream.next_batch(device=device)
            if batch is None:
                break
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=agent.config.autocast_enabled(device),
            ):
                loss, info, carry, previous_action = agent.loss(
                    batch,
                    carry,
                    previous_action,
                    update_norms=False,
                )
            count = batch.valid_records
            total_loss += float(loss.item()) * count
            total_records += count
            for key, value in info.items():
                totals[key] = totals.get(key, 0.0) + float(value.item()) * count
    agent.train(was_training)
    if total_records <= 0:
        raise RuntimeError(f"Validation dataset {dataset.path} produced no Dreamer sequences")
    return total_loss / total_records, {key: value / total_records for key, value in totals.items()}


def train_dreamer(config: dict, output_dir):
    del output_dir
    args = SimpleNamespace(**config)
    _validate_step_config(args)
    if str(args.obs_backend) != "state_vector":
        raise ValueError("Dreamer only supports obs_backend='state_vector'")
    set_seed(args.seed)
    device = get_torch_device()
    train_dataset, val_dataset = _load_datasets(args, "state_vector")
    model_config = DreamerConfig.from_mapping(
        config,
        obs_dim=train_dataset.obs_dim,
        num_actions=train_dataset.num_actions,
    )
    agent = DreamerAgent(model_config).to(device)
    optimizer = DreamerOptimizer(
        agent.optimized_parameters,
        learning_rate=float(getattr(args, "learning_rate", 4e-5)),
        agc=float(getattr(args, "dreamer_agc", 0.3)),
        eps=float(getattr(args, "dreamer_optimizer_eps", 1e-20)),
        beta1=float(getattr(args, "dreamer_beta1", 0.9)),
        beta2=float(getattr(args, "dreamer_beta2", 0.999)),
        warmup=int(getattr(args, "dreamer_warmup", 1000)),
    )
    sequence_length = int(getattr(args, "dreamer_batch_length", 64))
    stream = DreamerSequenceStream(
        train_dataset,
        batch_size=int(args.offline_batch_size),
        sequence_length=sequence_length,
        seed=int(args.seed),
        shuffle=True,
        cycle=True,
    )
    schedule = OfflineSchedule(
        total_steps=int(args.offline_train_steps),
        batch_size=int(args.offline_batch_size),
        total_sampled_transitions_estimate=int(args.offline_train_steps)
        * int(args.offline_batch_size)
        * sequence_length,
    )
    early_stopping = _make_early_stopping_state(args)
    train_history: list[dict] = []
    eval_history: list[dict] = []
    carry = None
    previous_action = None
    sampled_records = 0
    last_progress: OfflineProgress | None = None
    progress_bar = tqdm(
        range(1, schedule.total_steps + 1),
        desc=method_id(config),
        unit="step",
        dynamic_ncols=True,
        disable=not _use_tqdm_bars(args),
    )
    try:
        for step in progress_bar:
            batch = stream.next_batch(device=device)
            if batch is None:
                raise RuntimeError("Cycling Dreamer training stream was unexpectedly exhausted")
            sampled_records += batch.valid_records
            is_last = step == schedule.total_steps
            log_due = is_last or (_log_interval(args) > 0 and step % _log_interval(args) == 0)
            eval_due = _eval_during_training(args) and (
                is_last or (_eval_interval(args) > 0 and step % _eval_interval(args) == 0)
            )
            progress = OfflineProgress(step, sampled_records, log_due, eval_due, progress_bar)
            last_progress = progress
            if step == 1 or log_due:
                update_progress(
                    status="running",
                    phase="optimization",
                    method_id=method_id(config),
                    step=step,
                    total_steps=schedule.total_steps,
                    sampled_transitions=sampled_records,
                )
            agent.train()
            optimizer.zero_grad()
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=agent.config.autocast_enabled(device),
            ):
                loss, info, carry, previous_action = agent.loss(
                    batch,
                    carry,
                    previous_action,
                    update_norms=True,
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Dreamer produced non-finite loss at step {step}: {float(loss.item())}")
            loss.backward()
            optimizer_info = optimizer.step()
            agent.update_slow_value()
            # Recurrent state is intentionally carried across consecutive
            # chunks, but never across gradient graphs.
            carry = {key: value.detach() for key, value in carry.items()}
            previous_action = previous_action.detach()
            scalar_info = {key: float(value.item()) for key, value in info.items()}
            scalar_info.update(optimizer_info)
            _set_train_postfix(progress, float(loss.item()), {"wm": scalar_info["loss_rec"]})
            stop_due = False
            if log_due:
                val_loss, val_info = _dreamer_validation(agent, val_dataset, args, device)
                row = {
                    "step": step,
                    "sampled_transitions": sampled_records,
                    "loss": float(loss.item()),
                    "validation_loss": float(val_loss),
                    **scalar_info,
                    **{f"validation_{key}": value for key, value in val_info.items()},
                }
                train_history.append(row)
                _write_metrics(
                    args,
                    "val",
                    progress,
                    {
                        "train_loss": float(loss.item()),
                        "validation_loss": float(val_loss),
                        "validation_transitions": int(val_dataset.size),
                    },
                )
                checkpoint_improves = (
                    early_stopping.best_checkpoint is None
                    or float(val_loss) < early_stopping.best_loss - early_stopping.min_delta
                )
                checkpoint = _dreamer_checkpoint(agent) if checkpoint_improves else {}
                improved, stop_due = _update_early_stopping(
                    early_stopping,
                    validation_loss=float(val_loss),
                    progress=progress,
                    checkpoint=checkpoint,
                    checkpoint_is_snapshot=True,
                )
                _report_early_stopping(
                    args,
                    early_stopping,
                    progress=progress,
                    validation_loss=float(val_loss),
                    improved=improved,
                    stop_due=stop_due,
                )
            if eval_due:
                stats = _eval_stats(DreamerPolicy(agent), config, progress, args)
                eval_history.append(stats)
            if stop_due:
                break
    finally:
        progress_bar.close()

    training_metadata = _finalize_early_stopping(early_stopping, progress=last_progress, schedule=schedule)
    metrics = _metrics_from_history(
        train_history,
        eval_history,
        args,
        config,
        train_dataset,
        val_dataset,
        schedule,
        training_metadata=training_metadata,
    )
    return _final_result(metrics, _dreamer_checkpoint(agent), early_stopping.best_checkpoint)


OFFLINE_TRAINERS = {
    "raw_model": train_raw_model,
    "latent_model": train_latent_model,
    "reconstructive_model": train_reconstructive_model,
    "dqn": train_dqn,
    "bc": train_bc,
    "iql": train_iql,
    "cql": train_cql,
    "dreamer": train_dreamer,
}


def get_offline_trainer(algo: str):
    if algo not in OFFLINE_TRAINERS:
        raise KeyError(f"Unknown offline algorithm: {algo}")
    return OFFLINE_TRAINERS[algo]
