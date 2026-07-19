"""Select representative qualitative cases anchored on the initial state.

This script is designed for clinician-facing qualitative comparison of policies.
It avoids choosing anecdotal seeds or outcome-selected trajectories by:

1. Conditioning the analysis on the same initial state across models.
2. Requiring minimum support per model for a fixed horizon.
3. Summarizing each model by its modal sequence of actions/states.
4. Ranking candidate initial states by:
   coverage * disagreement * stability

Outputs:
- initial_state_candidates.csv: one row per eligible initial state
- selected_cases.csv: top-k initial states chosen for review
- case_sequences.csv: one row per selected case x model
- selected_cases.md: clinician-friendly summary
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
DEFAULT_DYNAMICS_PATH = (
    REPO_ROOT / "icu-sepsis" / "packages" / "icu_sepsis" / "icu_sepsis" / "envs" / "assets" / "dynamics.npz"
)

FEATURE_NAMES = [
    "gender",
    "mechvent",
    "max_dose_vaso",
    "re_admission",
    "age",
    "Weight_kg",
    "GCS",
    "HR",
    "SysBP",
    "MeanBP",
    "DiaBP",
    "RR",
    "Temp_C",
    "FiO2_1",
    "Potassium",
    "Sodium",
    "Chloride",
    "Glucose",
    "Magnesium",
    "Calcium",
    "Hb",
    "WBC_count",
    "Platelets_count",
    "PTT",
    "PT",
    "Arterial_pH",
    "paO2",
    "paCO2",
    "Arterial_BE",
    "HCO3",
    "Arterial_lactate",
    "SOFA",
    "SIRS",
    "Shock_Index",
    "PaO2_FiO2",
    "cumulated_balance",
    "SpO2",
    "BUN",
    "Creatinine",
    "SGOT",
    "SGPT",
    "Total_bili",
    "INR",
    "input_total",
    "input_4hourly",
    "output_total",
    "output_4hourly",
]

SPECIAL_FEATURES = {"max_dose_vaso"}
BINARY_FEATURES = {"gender", "mechvent", "re_admission"}
NORMAL_FEATURES = {
    "age", "Weight_kg", "GCS", "HR", "SysBP", "MeanBP", "DiaBP", "RR", "Temp_C",
    "FiO2_1", "Potassium", "Sodium", "Chloride", "Glucose", "Magnesium", "Calcium",
    "Hb", "WBC_count", "Platelets_count", "PTT", "PT", "Arterial_pH", "paO2", "paCO2",
    "Arterial_BE", "HCO3", "Arterial_lactate", "SOFA", "SIRS", "Shock_Index",
    "PaO2_FiO2", "cumulated_balance",
}
LOG_FEATURES = {
    "SpO2", "BUN", "Creatinine", "SGOT", "SGPT", "Total_bili", "INR",
    "input_total", "input_4hourly", "output_total", "output_4hourly",
}

CLINICAL_SUMMARY_FEATURES = [
    "GCS",
    "HR",
    "MeanBP",
    "RR",
    "FiO2_1",
    "Arterial_lactate",
    "Shock_Index",
    "PaO2_FiO2",
    "cumulated_balance",
    "BUN",
    "Creatinine",
    "input_4hourly",
    "output_4hourly",
]

TERMINAL_STATE_LABELS = {
    713: "terminal_death",
    714: "terminal_survival",
    715: "terminal_absorbing",
}

DISPLAY_NAMES = {
    "dqn_offline": "DQN",
    "ppo_logged_offline": "PPO",
    "raw_model_discrete_cem_offline": "Raw-D",
    "raw_model_continuous_cem_offline": "Raw-C",
    "latent_model_discrete_cem_offline": "Latent-D",
    "latent_model_continuous_cem_offline": "Latent-C",
    "reconstructive_model_discrete_cem_offline": "Reconstructive-D",
    "reconstructive_model_continuous_cem_offline": "Reconstructive-C",
}

DEFAULT_METHODS = [
    "dqn_offline",
    "ppo_logged_offline",
    "raw_model_discrete_cem_offline",
    "reconstructive_model_continuous_cem_offline",
]

DEFAULT_LENGTH_BUCKET_SIZE = 8

FLUID_BIN_LABELS = {
    0: "0 mL/4h",
    1: "1-50 mL/4h",
    2: "50-180 mL/4h",
    3: "180-530 mL/4h",
    4: ">530 mL/4h",
}

VASO_BIN_LABELS = {
    0: "0 mcg/kg/min",
    1: "0.001-0.08 mcg/kg/min",
    2: "0.08-0.22 mcg/kg/min",
    3: "0.22-0.45 mcg/kg/min",
    4: ">0.45 mcg/kg/min",
}


@dataclass(frozen=True)
class EpisodeSequence:
    eval_seed: int
    initial_state: int
    initial_sofa: float
    survived: bool
    action_seq: tuple[int, ...]
    state_seq: tuple[int, ...]
    length: int


@dataclass(frozen=True)
class ModalSequenceSummary:
    mode: tuple
    probability: float
    count: int


def _compute_optimal_policy(tx_mat: np.ndarray, r_mat: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    n_states = tx_mat.shape[0]
    v = np.zeros((1, 1, n_states), dtype=np.float64)
    for _ in range(50_000):
        tx_r = np.multiply(tx_mat, r_mat + gamma * v)
        v_new = np.max(np.sum(tx_r, axis=2), axis=1)
        if np.max(np.abs(v_new - v)) < 1e-6:
            v = v_new
            break
        v = v_new
    q_values = np.sum(np.multiply(tx_mat, r_mat + gamma * v), axis=2)
    return np.argmax(q_values, axis=1).astype(int)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifest",
        type=Path,
        help="Path to the run manifest JSON, e.g. runs/.../manifest.json",
    )
    parser.add_argument(
        "--dataset-tag",
        required=True,
        help="Dataset tag to analyze, e.g. 100k",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHODS,
        help="Method IDs from the manifest to compare.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=3,
        help="Number of actions/steps to summarize per case.",
    )
    parser.add_argument(
        "--min-support",
        type=int,
        default=30,
        help="Minimum number of episodes per model for a candidate initial state.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Number of initial states to select for qualitative review.",
    )
    parser.add_argument(
        "--length-bucket-size",
        type=int,
        default=DEFAULT_LENGTH_BUCKET_SIZE,
        help="Episode-length bucket width used to keep trajectories comparable.",
    )
    parser.add_argument(
        "--stratify-by-outcome",
        action="store_true",
        help="Require the same survived/failed outcome within each selected comparison cohort.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional explicit output directory. Defaults to reports/<run>/qualitative_cases/<dataset_tag>/",
    )
    parser.add_argument(
        "--dynamics-path",
        type=Path,
        default=DEFAULT_DYNAMICS_PATH,
        help="Path to icu-sepsis dynamics.npz with state centroids and SOFA scores.",
    )
    return parser.parse_args()


def _load_manifest(manifest_path: Path) -> list[dict]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Manifest must contain a list of rows: {manifest_path}")
    return payload


def _resolve_output_dir(manifest_path: Path, dataset_tag: str, output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir
    run_name = manifest_path.parent.name
    return PROJECT_ROOT / "reports" / run_name / "qualitative_cases" / dataset_tag


def _load_dynamics(dynamics_path: Path) -> dict[str, np.ndarray]:
    arrays = np.load(dynamics_path)
    return {name: arrays[name] for name in arrays.files}


def _filter_manifest_rows(
    manifest_rows: list[dict],
    dataset_tag: str,
    methods: list[str],
) -> list[dict]:
    selected: list[dict] = []
    for method_id in methods:
        matches = [
            row for row in manifest_rows
            if row.get("dataset_tag") == dataset_tag and row.get("method_id") == method_id
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one manifest row for dataset_tag={dataset_tag!r}, "
                f"method_id={method_id!r}; found {len(matches)}"
            )
        selected.append(matches[0])
    return selected


def _load_trajectories(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"eval_seed", "t", "state_id", "sofa_score", "action_idx", "reward", "done", "survived"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Trajectories file {path} is missing columns: {sorted(missing)}")
    return df.sort_values(["eval_seed", "t"]).reset_index(drop=True)


def _terminal_state_id(row: pd.Series) -> int:
    survived = bool(row["survived"])
    reward = float(row["reward"])
    if survived or reward > 0.0:
        return 714
    return 713


def build_episode_sequences(df: pd.DataFrame, horizon: int) -> list[EpisodeSequence]:
    sequences: list[EpisodeSequence] = []
    for eval_seed, episode in df.groupby("eval_seed", sort=False):
        episode = episode.sort_values("t").reset_index(drop=True)
        if len(episode) < horizon:
            continue
        action_seq = tuple(int(value) for value in episode.loc[: horizon - 1, "action_idx"].tolist())
        state_seq: list[int] = [int(episode.loc[0, "state_id"])]
        for step in range(horizon):
            if step + 1 < len(episode):
                next_state = int(episode.loc[step + 1, "state_id"])
            else:
                next_state = _terminal_state_id(episode.loc[step])
            state_seq.append(next_state)
        sequences.append(
            EpisodeSequence(
                eval_seed=int(eval_seed),
                initial_state=int(episode.loc[0, "state_id"]),
                initial_sofa=float(episode.loc[0, "sofa_score"]),
                survived=bool(episode.loc[len(episode) - 1, "survived"]),
                action_seq=action_seq,
                state_seq=tuple(state_seq),
                length=int(len(episode)),
            )
        )
    return sequences


def _mode_summary(values: Iterable[tuple]) -> ModalSequenceSummary:
    counts = Counter(values)
    if not counts:
        return ModalSequenceSummary(mode=tuple(), probability=0.0, count=0)
    mode, count = counts.most_common(1)[0]
    total = sum(counts.values())
    return ModalSequenceSummary(mode=mode, probability=float(count / total), count=int(count))


def _distribution(values: Iterable[int], size: int) -> np.ndarray:
    arr = np.zeros(size, dtype=np.float64)
    total = 0
    for value in values:
        idx = int(value)
        if 0 <= idx < size:
            arr[idx] += 1.0
            total += 1
    if total > 0:
        arr /= total
    return arr


def normalized_jsd(distributions: list[np.ndarray]) -> float:
    if not distributions:
        return 0.0
    matrix = np.stack(distributions, axis=0).astype(np.float64)
    row_sums = matrix.sum(axis=1, keepdims=True)
    valid = row_sums.squeeze(-1) > 0
    if not np.any(valid):
        return 0.0
    matrix = matrix[valid]
    row_sums = matrix.sum(axis=1, keepdims=True)
    matrix = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums > 0)
    mixture = matrix.mean(axis=0)

    def entropy(prob: np.ndarray) -> float:
        prob = prob[prob > 0]
        if prob.size == 0:
            return 0.0
        return float(-(prob * np.log2(prob)).sum())

    jsd = entropy(mixture) - float(np.mean([entropy(row) for row in matrix]))
    dimension = mixture.size
    if dimension <= 1:
        return 0.0
    return float(max(0.0, jsd / math.log2(dimension)))


def _action_label(action_idx: int) -> str:
    fluid_bin = action_idx // 5
    vaso_bin = action_idx % 5
    return f"{FLUID_BIN_LABELS[fluid_bin]} + {VASO_BIN_LABELS[vaso_bin]}"


def _format_action_sequence(action_seq: tuple[int, ...]) -> str:
    return " -> ".join(_action_label(action_idx) for action_idx in action_seq)


def _format_state_sequence_ids(state_seq: tuple[int, ...]) -> str:
    return " -> ".join(TERMINAL_STATE_LABELS.get(state_id, str(state_id)) for state_id in state_seq)


def _format_state_sequence_sofa(state_seq: tuple[int, ...], sofa_scores: np.ndarray) -> str:
    labels: list[str] = []
    for state_id in state_seq:
        if state_id in TERMINAL_STATE_LABELS:
            labels.append(TERMINAL_STATE_LABELS[state_id])
        else:
            labels.append(f"{state_id} (SOFA {sofa_scores[state_id]:.2f})")
    return " -> ".join(labels)


def _step_rows_from_joint_mode(
    action_seq: tuple[int, ...],
    state_seq: tuple[int, ...],
    sofa_scores: np.ndarray,
    optimal_policy: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for step_idx, action_idx in enumerate(action_seq):
        current_state = int(state_seq[step_idx])
        next_state = int(state_seq[step_idx + 1])
        optimal_action_idx = None if current_state in TERMINAL_STATE_LABELS else int(optimal_policy[current_state])
        rows.append(
            {
                "step": step_idx,
                "current_state": current_state,
                "current_state_label": TERMINAL_STATE_LABELS.get(current_state, str(current_state)),
                "current_state_sofa": None if current_state in TERMINAL_STATE_LABELS else float(sofa_scores[current_state]),
                "action_idx": int(action_idx),
                "action_label": _action_label(int(action_idx)),
                "optimal_action_idx": optimal_action_idx,
                "optimal_action_label": "" if optimal_action_idx is None else _action_label(optimal_action_idx),
                "next_state": next_state,
                "next_state_label": TERMINAL_STATE_LABELS.get(next_state, str(next_state)),
                "next_state_sofa": None if next_state in TERMINAL_STATE_LABELS else float(sofa_scores[next_state]),
            }
        )
    return rows


def _qualitative_band(value: float) -> str:
    if value >= 2.0:
        return "muito alta"
    if value >= 1.0:
        return "alta"
    if value >= 0.4:
        return "levemente alta"
    if value <= -2.0:
        return "muito baixa"
    if value <= -1.0:
        return "baixa"
    if value <= -0.4:
        return "levemente baixa"
    return "na media"


def _feature_type(feature_name: str) -> str:
    if feature_name in BINARY_FEATURES:
        return "binary_centered"
    if feature_name in SPECIAL_FEATURES:
        return "special"
    if feature_name in NORMAL_FEATURES:
        return "zscore"
    if feature_name in LOG_FEATURES:
        return "log_zscore"
    return "unknown"


def _feature_type_label(feature_name: str) -> str:
    feature_type = _feature_type(feature_name)
    if feature_type == "binary_centered":
        return "binaria"
    if feature_type == "special":
        return "especial"
    if feature_type == "zscore":
        return "continua"
    if feature_type == "log_zscore":
        return "continua"
    return "outro"


def _binary_prevalence_label(probability: float) -> str:
    if probability <= 0.1:
        return "valor 1 muito raro neste cluster"
    if probability <= 0.3:
        return "valor 1 incomum neste cluster"
    if probability < 0.7:
        return "cluster misto"
    if probability < 0.9:
        return "valor 1 comum neste cluster"
    return "valor 1 muito comum neste cluster"


def _feature_interpretation(feature_name: str, value: float) -> tuple[str, str]:
    feature_type = _feature_type(feature_name)
    if feature_type == "binary_centered":
        probability = float(min(1.0, max(0.0, value + 0.5)))
        return (
            "valor bruto exato indisponivel; prevalencia estimada do valor 1 no cluster",
            f"p≈{probability:.2f}; {_binary_prevalence_label(probability)}",
        )
    if feature_type == "special":
        return (
            "valor bruto exato indisponivel no env publico",
            "feature especial; usar apenas como sinal relativo, nao como unidade clinica recuperavel",
        )
    if feature_type in {"zscore", "log_zscore"}:
        return (
            "valor bruto exato indisponivel no env publico",
            _qualitative_band(value),
        )
    return (
        "valor bruto exato indisponivel no env publico",
        "apenas sinal relativo",
    )


def _feature_clinician_label(feature_name: str, value: float) -> str:
    feature_type = _feature_type(feature_name)
    if feature_type == "binary_centered":
        probability = float(min(1.0, max(0.0, value + 0.5)))
        return _binary_prevalence_label(probability)
    if feature_type == "special":
        return "usar apenas como sinal relativo"
    return _qualitative_band(value)


def _chunked_feature_rows(center: np.ndarray, n_chunks: int = 3) -> list[list[tuple[str, str, str]]]:
    rows = [
        (
            feature_name,
            _feature_type_label(feature_name),
            _feature_clinician_label(feature_name, float(value)),
        )
        for feature_name, value in zip(FEATURE_NAMES, center)
    ]
    chunk_size = math.ceil(len(rows) / n_chunks)
    return [rows[idx:idx + chunk_size] for idx in range(0, len(rows), chunk_size)]


def _render_feature_tables_html(center: np.ndarray) -> list[str]:
    chunks = _chunked_feature_rows(center, n_chunks=3)
    lines = [
        '<table>',
        '<tr>',
    ]
    for chunk in chunks:
        lines.append('<td valign="top">')
        lines.append('<table>')
        lines.append('<thead><tr><th>Feature</th><th>Tipo</th><th>Situacao do cluster</th></tr></thead>')
        lines.append('<tbody>')
        for feature_name, feature_type, feature_label in chunk:
            lines.append(
                f'<tr><td>{feature_name}</td><td>{feature_type}</td><td>{feature_label}</td></tr>'
            )
        lines.append('</tbody>')
        lines.append('</table>')
        lines.append('</td>')
    lines.extend([
        '</tr>',
        '</table>',
    ])
    return lines


def _state_summary_text(initial_state: int, centers: np.ndarray) -> str:
    row = centers[initial_state]
    entries: list[tuple[str, float]] = []
    for feature in CLINICAL_SUMMARY_FEATURES:
        if feature in FEATURE_NAMES:
            idx = FEATURE_NAMES.index(feature)
            if 0 <= idx < row.size:
                entries.append((feature, float(row[idx])))
            else:
                entries.append((feature, 0.0))
        else:
            entries.append((feature, 0.0))
    entries.sort(key=lambda item: abs(item[1]), reverse=True)
    return "; ".join(
        f"{feature} {_qualitative_band(value)}"
        for feature, value in entries[:4]
    )


def _feature_columns(initial_state: int, centers: np.ndarray) -> dict[str, float]:
    row = centers[initial_state]
    result: dict[str, float] = {}
    for feature in CLINICAL_SUMMARY_FEATURES:
        if feature in FEATURE_NAMES:
            idx = FEATURE_NAMES.index(feature)
            if 0 <= idx < row.size:
                result[f"feature_{feature}"] = float(row[idx])
            else:
                result[f"feature_{feature}"] = 0.0
        else:
            result[f"feature_{feature}"] = 0.0
    return result


def _expert_top_action(initial_state: int, expert_policy: np.ndarray) -> tuple[int, float]:
    distribution = expert_policy[initial_state]
    action_idx = int(np.argmax(distribution))
    return action_idx, float(distribution[action_idx])


def _normalize_series(series: pd.Series) -> pd.Series:
    max_value = float(series.max()) if not series.empty else 0.0
    if max_value <= 0.0:
        return pd.Series(np.zeros(len(series), dtype=np.float64), index=series.index)
    return series.astype(np.float64) / max_value


def _length_bucket(length: int, bucket_size: int) -> int:
    if bucket_size <= 0:
        raise ValueError("length bucket size must be positive")
    if length <= 0:
        return 0
    return int((length - 1) // bucket_size)


def _length_bucket_bounds(length_bucket: int, bucket_size: int) -> tuple[int, int]:
    lower = length_bucket * bucket_size + 1
    upper = (length_bucket + 1) * bucket_size
    return lower, upper


def _joint_signature(action_seq: tuple[int, ...], state_seq: tuple[int, ...]) -> str:
    action_text = ",".join(str(value) for value in action_seq)
    state_text = ",".join(str(value) for value in state_seq)
    return f"{action_text}::{state_text}"


def _joint_signature_distance(first: tuple[int, ...], second: tuple[int, ...]) -> int:
    limit = min(len(first), len(second))
    distance = sum(1 for idx in range(limit) if first[idx] != second[idx])
    distance += abs(len(first) - len(second))
    return int(distance)


def summarize_initial_states(
    sequences_by_method: dict[str, list[EpisodeSequence]],
    methods: list[str],
    horizon: int,
    min_support: int,
    length_bucket_size: int,
    stratify_by_outcome: bool,
    centers: np.ndarray,
    sofa_scores: np.ndarray,
    expert_policy: np.ndarray,
    optimal_policy: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_method_frames = {
        method_id: pd.DataFrame(
            {
                "eval_seed": [seq.eval_seed for seq in sequences],
                "initial_state": [seq.initial_state for seq in sequences],
                "initial_sofa": [seq.initial_sofa for seq in sequences],
                "survived": [seq.survived for seq in sequences],
                "action_seq": [seq.action_seq for seq in sequences],
                "state_seq": [seq.state_seq for seq in sequences],
                "length": [seq.length for seq in sequences],
                "length_bucket": [_length_bucket(seq.length, length_bucket_size) for seq in sequences],
            }
        )
        for method_id, sequences in sequences_by_method.items()
    }

    candidate_keys = sorted(
        set.intersection(
            *[
                set(
                    tuple(row)
                    for row in frame[["initial_state", "length_bucket"] + (["survived"] if stratify_by_outcome else [])].drop_duplicates().itertuples(index=False, name=None)
                )
                for frame in per_method_frames.values()
            ]
        )
    )
    candidate_rows: list[dict] = []
    sequence_rows: list[dict] = []

    for candidate_key in candidate_keys:
        initial_state = int(candidate_key[0])
        length_bucket = int(candidate_key[1])
        outcome = bool(candidate_key[2]) if stratify_by_outcome else None
        # buffer rows for this candidate and only commit if candidate passes gating
        local_sequence_rows: list[dict] = []
        state_frames = {
            method_id: frame[
                (frame["initial_state"] == initial_state)
                & (frame["length_bucket"] == length_bucket)
                & ((frame["survived"] == outcome) if stratify_by_outcome else True)
            ].copy()
            for method_id, frame in per_method_frames.items()
        }
        supports = {method_id: int(len(frame)) for method_id, frame in state_frames.items()}
        if min(supports.values()) < min_support:
            continue

        step_jsd: list[float] = []
        for step in range(horizon):
            distributions = [
                _distribution(frame["action_seq"].map(lambda seq, idx=step: seq[idx]), size=25)
                for frame in state_frames.values()
            ]
            step_jsd.append(normalized_jsd(distributions))
        disagreement_score = float(np.mean(step_jsd)) if step_jsd else 0.0

        model_stabilities: list[float] = []
        modal_step0_actions: dict[str, int] = {}
        modal_joint_signatures: list[str] = []
        modal_joint_sequences: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        for method_id, frame in state_frames.items():
            action_mode = _mode_summary(frame["action_seq"].tolist())
            state_mode = _mode_summary(frame["state_seq"].tolist())
            joint_mode = _mode_summary(list(zip(frame["action_seq"].tolist(), frame["state_seq"].tolist())))
            stability = 0.5 * (action_mode.probability + state_mode.probability)
            model_stabilities.append(stability)
            modal_step0_actions[method_id] = int(action_mode.mode[0]) if action_mode.mode else -1
            if joint_mode.mode:
                modal_joint_sequences.append((tuple(joint_mode.mode[0]), tuple(joint_mode.mode[1])))
                modal_joint_signatures.append(_joint_signature(tuple(joint_mode.mode[0]), tuple(joint_mode.mode[1])))

            local_sequence_rows.append(
                {
                    "initial_state": initial_state,
                    "length_bucket": length_bucket,
                    "length_bucket_min": _length_bucket_bounds(length_bucket, length_bucket_size)[0],
                    "length_bucket_max": _length_bucket_bounds(length_bucket, length_bucket_size)[1],
                    "survived": outcome if stratify_by_outcome else None,
                    "method_id": method_id,
                    "method_label": DISPLAY_NAMES.get(method_id, method_id),
                    "support": supports[method_id],
                    "modal_step0_action": int(action_mode.mode[0]) if action_mode.mode else -1,
                    "modal_step0_action_label": _action_label(int(action_mode.mode[0])) if action_mode.mode else "",
                    "modal_action_seq_ids": "|".join(str(value) for value in action_mode.mode),
                    "modal_action_seq_labels": _format_action_sequence(action_mode.mode),
                    "modal_action_seq_prob": action_mode.probability,
                    "modal_state_seq_ids": "|".join(str(value) for value in state_mode.mode),
                    "modal_state_seq_labels": _format_state_sequence_ids(state_mode.mode),
                    "modal_state_seq_sofa": _format_state_sequence_sofa(state_mode.mode, sofa_scores),
                    "modal_state_seq_prob": state_mode.probability,
                    "modal_joint_prob": joint_mode.probability,
                    "modal_joint_count": joint_mode.count,
                    "modal_joint_action_seq_ids": "|".join(str(value) for value in joint_mode.mode[0]),
                    "modal_joint_action_seq_labels": _format_action_sequence(joint_mode.mode[0]),
                    "modal_joint_state_seq_ids": "|".join(str(value) for value in joint_mode.mode[1]),
                    "modal_joint_state_seq_labels": _format_state_sequence_ids(joint_mode.mode[1]),
                    "modal_joint_state_seq_sofa": _format_state_sequence_sofa(joint_mode.mode[1], sofa_scores),
                    "modal_joint_steps_json": json.dumps(
                        _step_rows_from_joint_mode(
                            joint_mode.mode[0],
                            joint_mode.mode[1],
                            sofa_scores,
                            optimal_policy,
                        ),
                        ensure_ascii=True,
                    ),
                    "stability_score": stability,
                }
            )

        stability_score = float(np.mean(model_stabilities)) if model_stabilities else 0.0
        unique_modal_joint_sequences = len(set(modal_joint_signatures))
        if unique_modal_joint_sequences < 2:
            continue
        modal_joint_distances = [
            _joint_signature_distance(first[0], second[0]) + _joint_signature_distance(first[1], second[1])
            for idx, first in enumerate(modal_joint_sequences)
            for second in modal_joint_sequences[idx + 1 :]
        ]
        if modal_joint_distances and max(modal_joint_distances) <= 0:
            continue
        expert_action, expert_prob = _expert_top_action(initial_state, expert_policy)
        row = {
            "candidate_key": f"{initial_state}:{length_bucket}" + (f":{int(outcome)}" if stratify_by_outcome else ""),
            "initial_state": initial_state,
            "length_bucket": length_bucket,
            "length_bucket_min": _length_bucket_bounds(length_bucket, length_bucket_size)[0],
            "length_bucket_max": _length_bucket_bounds(length_bucket, length_bucket_size)[1],
            "survived": outcome if stratify_by_outcome else None,
            "initial_sofa": float(sofa_scores[initial_state]),
            "coverage_raw": min(supports.values()),
            "coverage_mean": float(np.mean(list(supports.values()))),
            "disagreement_step_0": step_jsd[0] if len(step_jsd) > 0 else 0.0,
            "disagreement_step_1": step_jsd[1] if len(step_jsd) > 1 else 0.0,
            "disagreement_step_2": step_jsd[2] if len(step_jsd) > 2 else 0.0,
            "disagreement_score": disagreement_score,
            "stability_score": stability_score,
            "unique_modal_joint_sequences": unique_modal_joint_sequences,
            "valid_comparison": True,
            "state_summary": _state_summary_text(initial_state, centers),
            "expert_top_action": expert_action,
            "expert_top_action_label": _action_label(expert_action),
            "expert_top_action_prob": expert_prob,
        }
        for method_id, support in supports.items():
            label = DISPLAY_NAMES.get(method_id, method_id)
            row[f"support_{label}"] = support
            row[f"modal_step0_action_{label}"] = modal_step0_actions[method_id]
            row[f"modal_step0_action_label_{label}"] = _action_label(modal_step0_actions[method_id])
        row.update(_feature_columns(initial_state, centers))
        # commit the per-method rows for this candidate only if the candidate passed gating
        sequence_rows.extend(local_sequence_rows)
        candidate_rows.append(row)

    candidates = pd.DataFrame(candidate_rows)
    sequences = pd.DataFrame(sequence_rows)
    if candidates.empty:
        return candidates, sequences

    candidates = candidates.sort_values(["coverage_raw", "disagreement_score"], ascending=[False, False]).reset_index(drop=True)
    candidates = candidates.drop_duplicates(subset=["initial_state"], keep="first").reset_index(drop=True)
    candidates["coverage_score"] = _normalize_series(candidates["coverage_raw"])
    candidates["representativeness_score"] = candidates["coverage_score"] * candidates["stability_score"]
    candidates["divergence_score"] = candidates["coverage_score"] * candidates["disagreement_score"]
    candidates["selection_score"] = (
        candidates["coverage_score"] * candidates["disagreement_score"] * candidates["stability_score"]
    )
    candidates = candidates.sort_values(
        ["selection_score", "coverage_raw", "disagreement_score", "stability_score"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    candidates["case_rank"] = np.arange(1, len(candidates) + 1)
    rank_map = dict(zip(candidates["candidate_key"], candidates["case_rank"]))
    if not sequences.empty:
        sequences["candidate_key"] = sequences.apply(
            lambda row: f"{int(row['initial_state'])}:{int(row['length_bucket'])}" + (f":{int(row['survived'])}" if stratify_by_outcome else ""),
            axis=1,
        )
        sequences = sequences[sequences["candidate_key"].isin(candidates["candidate_key"])]
        sequences["case_rank"] = sequences["candidate_key"].map(rank_map)
        sequences = sequences.sort_values(["case_rank", "method_label"]).reset_index(drop=True)
    return candidates, sequences


def _select_case_mix(candidates: pd.DataFrame, top_k: int) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    representative_k = max(1, top_k // 2)
    divergent_k = max(0, top_k - representative_k)

    representative = (
        candidates.sort_values(
            ["representativeness_score", "coverage_raw", "stability_score", "disagreement_score"],
            ascending=[False, False, False, False],
        )
        .head(representative_k)
        .copy()
    )
    representative["selection_bucket"] = "representative"
    representative["bucket_rank"] = np.arange(1, len(representative) + 1)

    remaining = candidates[~candidates["initial_state"].isin(representative["initial_state"])].copy()
    divergent = (
        remaining.sort_values(
            ["divergence_score", "coverage_raw", "disagreement_score", "stability_score"],
            ascending=[False, False, False, False],
        )
        .head(divergent_k)
        .copy()
    )
    divergent["selection_bucket"] = "divergent"
    divergent["bucket_rank"] = np.arange(1, len(divergent) + 1)

    selected = pd.concat([representative, divergent], ignore_index=True)
    selected = selected.sort_values(["selection_bucket", "bucket_rank"], ascending=[True, True]).reset_index(drop=True)
    selected = selected.drop_duplicates(subset=["initial_state"], keep="first").reset_index(drop=True)
    selected["case_rank"] = np.arange(1, len(selected) + 1)
    return selected


def _build_review_payload(selected: pd.DataFrame, sequence_rows: pd.DataFrame, methods: list[str]) -> dict[str, object]:
    payload_cases: list[dict[str, object]] = []
    for _, case in selected.iterrows():
        case_key = case["candidate_key"]
        case_sequences = sequence_rows[sequence_rows["candidate_key"] == case_key].copy()
        method_rows: list[dict[str, object]] = []
        for _, row in case_sequences.iterrows():
            method_rows.append(
                {
                    "method_id": row["method_id"],
                    "method_label": row["method_label"],
                    "support": int(row["support"]),
                    "modal_step0_action": int(row["modal_step0_action"]),
                    "modal_step0_action_label": row["modal_step0_action_label"],
                    "modal_action_seq_ids": row["modal_action_seq_ids"],
                    "modal_action_seq_labels": row["modal_action_seq_labels"],
                    "modal_state_seq_ids": row["modal_state_seq_ids"],
                    "modal_state_seq_labels": row["modal_state_seq_labels"],
                    "modal_state_seq_sofa": row["modal_state_seq_sofa"],
                    "modal_joint_prob": float(row["modal_joint_prob"]),
                    "modal_joint_count": int(row["modal_joint_count"]),
                    "stability_score": float(row["stability_score"]),
                    "steps": json.loads(row["modal_joint_steps_json"]),
                }
            )
        payload_cases.append(
            {
                "case_rank": int(case["case_rank"]),
                "candidate_key": case_key,
                "initial_state": int(case["initial_state"]),
                "length_bucket": int(case["length_bucket"]),
                "length_bucket_min": int(case["length_bucket_min"]),
                "length_bucket_max": int(case["length_bucket_max"]),
                "survived": None if pd.isna(case["survived"]) else bool(case["survived"]),
                "selection_bucket": case["selection_bucket"],
                "coverage_raw": int(case["coverage_raw"]),
                "disagreement_score": float(case["disagreement_score"]),
                "stability_score": float(case["stability_score"]),
                "selection_score": float(case["selection_score"]),
                "unique_modal_joint_sequences": int(case["unique_modal_joint_sequences"]),
                "valid_comparison": bool(case["valid_comparison"]),
                "state_summary": case["state_summary"],
                "methods": method_rows,
            }
        )
    return {
        "version": 1,
        "methods": methods,
        "cases": payload_cases,
    }


def _write_markdown(
    path: Path,
    selected: pd.DataFrame,
    sequence_rows: pd.DataFrame,
    methods: list[str],
    centers: np.ndarray,
    include_optimal_actions: bool = True,
) -> None:
    method_labels = [DISPLAY_NAMES.get(method_id, method_id) for method_id in methods]
    lines: list[str] = []
    lines.append("# Casos Qualitativos Selecionados")
    lines.append("")
    lines.append("## O que significam as acoes")
    lines.append("")
    lines.append("- **Fluido**: volume total de fluidos intravenosos no bloco de 4 horas.")
    lines.append("- No paper, isso inclui **cristaloides, coloides e hemoderivados**, normalizados por tonicidade.")
    lines.append("- **Vasopressor**: dose maxima no bloco de 4 horas.")
    lines.append("- No paper, isso agrega **norepinefrina, epinefrina, vasopressina, dopamina e fenilefrina**, convertidos para **equivalente de norepinefrina**.")
    lines.append("- Unidade do vasopressor nas acoes: `mcg/kg/min`.")
    lines.append("- Unidade do fluido nas acoes: `mL/4h`.")
    lines.append("")
    lines.append("### Bins de vasopressor")
    lines.append("")
    lines.append("- `0`: sem vasopressor")
    lines.append("- `1`: `0.001-0.08 mcg/kg/min`")
    lines.append("- `2`: `0.08-0.22 mcg/kg/min`")
    lines.append("- `3`: `0.22-0.45 mcg/kg/min`")
    lines.append("- `4`: `>0.45 mcg/kg/min`")
    lines.append("")
    lines.append("### Bins de fluido")
    lines.append("")
    lines.append("- `0`: `0 mL/4h`")
    lines.append("- `1`: `1-50 mL/4h`")
    lines.append("- `2`: `50-180 mL/4h`")
    lines.append("- `3`: `180-530 mL/4h`")
    lines.append("- `4`: `>530 mL/4h`")
    lines.append("")
    lines.append("## Tradução rápida de algumas medidas")
    lines.append("")
    lines.append("- `HR`: frequência cardíaca")
    lines.append("- `SysBP`: pressão sistólica")
    lines.append("- `MeanBP`: pressão arterial média")
    lines.append("- `DiaBP`: pressão diastólica")
    lines.append("- `RR`: frequência respiratória")
    lines.append("- `FiO2_1`: fração inspirada de oxigênio")
    lines.append("- `GCS`: escala de coma de Glasgow")
    lines.append("- `Arterial_lactate`: lactato arterial")
    lines.append("- `PaO2_FiO2`: relação PaO2/FiO2")
    lines.append("- `BUN`: ureia")
    lines.append("- `Creatinine`: creatinina")
    lines.append("- `SGOT`/`SGPT`: transaminases")
    lines.append("- `INR`: coagulação")
    lines.append("- `input_4hourly`: fluido recebido nas últimas 4 horas")
    lines.append("- `output_4hourly`: diurese/saída nas últimas 4 horas")
    lines.append("- `cumulated_balance`: balanço hídrico acumulado")
    lines.append("- `SIRS`: critérios inflamatórios sistêmicos")
    lines.append("- `Shock_Index`: índice de choque")
    lines.append("")
    lines.append("## Como os casos foram escolhidos")
    lines.append("")
    lines.append("1. A unidade de seleção é o `estado inicial`, não um episódio isolado.")
    lines.append("2. Um estado só entra se todos os modelos tiverem pelo menos `min_support` episódios com horizonte de 3 ações a partir dele.")
    lines.append("3. Para cada modelo e estado inicial, resumimos o comportamento pela `trajetória modal conjunta` de 3 passos:")
    lines.append("   ação no passo 0 -> próximo estado -> ação no passo 1 -> próximo estado -> ação no passo 2 -> próximo estado.")
    lines.append("4. Cada estado recebe três medidas:")
    lines.append("   `coverage_score`: quão frequente ele é;")
    lines.append("   `disagreement_score`: quão diferentes são as distribuições de ações dos modelos;")
    lines.append("   `stability_score`: quão concentradas são as sequências modais.")
    lines.append("5. A seleção final mistura dois grupos:")
    lines.append("   `representative`: estados frequentes e estáveis;")
    lines.append("   `divergent`: estados frequentes onde os modelos mais discordam.")
    lines.append("")
    lines.append("## Como ler as features")
    lines.append("")
    lines.append("- O vetor abaixo é o centróide do cluster no espaço **normalizado** do `icu-sepsis`.")
    lines.append("- Essa normalizacao foi feita **antes do clustering**, usando o conjunto inteiro de registros do cohort usado para construir o MDP.")
    lines.append("- Portanto, `alta`, `baixa` e `na media` significam **em relacao ao cohort inteiro**, e nao em relacao aos outros pontos do mesmo cluster.")
    lines.append("- Como todos os clusters vivem no mesmo espaco normalizado global, **comparar clusters entre si faz sentido** em termos relativos.")
    lines.append("- O env publicado **não inclui** os parâmetros de desnormalização (`mu/sigma`) nem o cohort bruto usado para construir os clusters.")
    lines.append("- Por isso, para as features contínuas, **não temos como recuperar o valor clínico exato** a partir do `state_vector` sozinho.")
    lines.append("- O código original do `AI_Clinician` calcula `cmu/csigma` e `dmu/dsigma` a partir do `MIMICraw`, mas essas estatísticas não são publicadas no benchmark final.")
    lines.append("- Cada valor é, portanto, um z-score, log-zscore ou valor centrado, não uma unidade clínica bruta.")
    lines.append("- `0` significa perto da média do cohort; positivo significa acima da média; negativo, abaixo.")
    lines.append("- A decisão clínica deve usar esse vetor como descrição relativa do fenótipo, não como medida real de beira-leito.")
    lines.append("- Para `gender`, `mechvent` e `re_admission`, a melhor aproximação disponível é a prevalência estimada do valor `1` no cluster: `p ≈ valor_normalizado + 0.5`.")
    lines.append("- Se você tiver acesso ao cohort bruto intermediário usado na criação do MDP, aí sim dá para inverter exatamente:")
    lines.append("  `normal: x = z * sigma + mu`; `log: x = exp(z * sigma + mu) - 0.1`.")
    lines.append("")
    lines.append(
        "Seleção mista: metade dos casos prioriza `coverage_score * stability_score` "
        "(representatividade) e metade prioriza `coverage_score * disagreement_score` "
        "(divergência)."
    )
    lines.append("")
    for _, case in selected.iterrows():
        lines.append(f"## Caso {int(case['case_rank'])}: estado inicial {int(case['initial_state'])}")
        lines.append("")
        lines.append(f"- Grupo de seleção: `{case['selection_bucket']}`")
        lines.append(f"- Chave do cohort: `{case['candidate_key']}`")
        lines.append(f"- Faixa de comprimento: `{int(case['length_bucket_min'])}-{int(case['length_bucket_max'])}` passos")
        lines.append(f"- SOFA inicial: `{case['initial_sofa']:.2f}`")
        lines.append(f"- Cobertura mínima por modelo: `{int(case['coverage_raw'])}` episódios")
        lines.append(f"- Desacordo médio entre modelos: `{case['disagreement_score']:.3f}`")
        lines.append(f"- Estabilidade média: `{case['stability_score']:.3f}`")
        lines.append(f"- Score final: `{case['selection_score']:.3f}`")
        lines.append(f"- Resumo clínico: {case['state_summary']}")
        lines.append(f"- Referência expert: `{int(case['expert_top_action'])}` = {case['expert_top_action_label']}")
        lines.append("")
        lines.append("### Vetor completo do estado inicial")
        lines.append("")
        center = centers[int(case["initial_state"])]
        lines.extend(_render_feature_tables_html(center))
        lines.append("")
        lines.append("### Sequência modal conjunta de cada modelo")
        lines.append("")
        case_rows = sequence_rows[sequence_rows["candidate_key"] == case["candidate_key"]]
        case_rows = case_rows.set_index("method_label")
        for label in method_labels:
            row = case_rows.loc[label]
            lines.append(f"#### {label}")
            lines.append("")
            lines.append(f"- Suporte: `{int(row['support'])}` episódios")
            lines.append(f"- Probabilidade da trajetória modal conjunta: `{row['modal_joint_prob']:.3f}`")
            lines.append(f"- Ações modais isoladas: `{row['modal_action_seq_labels']}` (prob. `{row['modal_action_seq_prob']:.3f}`)")
            lines.append(f"- Estados modais isolados: `{row['modal_state_seq_sofa']}` (prob. `{row['modal_state_seq_prob']:.3f}`)")
            lines.append("")
            if include_optimal_actions:
                lines.append("| Passo | Estado atual | SOFA atual | Ação do modelo | Descrição da ação do modelo | Próximo estado | SOFA próximo | Ação ótima do MDP | Descrição da ação ótima |")
                lines.append("| ---: | --- | ---: | --- | --- | --- | ---: | --- | --- |")
            else:
                lines.append("| Passo | Estado atual | SOFA atual | Ação do modelo | Descrição da ação do modelo | Próximo estado | SOFA próximo |")
                lines.append("| ---: | --- | ---: | --- | --- | --- | ---: |")
            step_rows = json.loads(row["modal_joint_steps_json"])
            for step_row in step_rows:
                current_sofa = "" if step_row["current_state_sofa"] is None else f"{float(step_row['current_state_sofa']):.2f}"
                next_sofa = "" if step_row["next_state_sofa"] is None else f"{float(step_row['next_state_sofa']):.2f}"
                optimal_idx = "" if step_row["optimal_action_idx"] is None else int(step_row["optimal_action_idx"])
                if include_optimal_actions:
                    lines.append(
                        f"| {int(step_row['step'])} | {step_row['current_state_label']} | {current_sofa} | "
                        f"{int(step_row['action_idx'])} | {step_row['action_label']} | "
                        f"{step_row['next_state_label']} | {next_sofa} | "
                        f"{optimal_idx} | {step_row['optimal_action_label']} |"
                    )
                else:
                    lines.append(
                        f"| {int(step_row['step'])} | {step_row['current_state_label']} | {current_sofa} | "
                        f"{int(step_row['action_idx'])} | {step_row['action_label']} | "
                        f"{step_row['next_state_label']} | {next_sofa} |"
                    )
            lines.append("")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _run_generic_selection(args: argparse.Namespace, selected_rows: list[dict], output_dir: Path) -> dict[str, Path]:
    """Create qualitative artifacts from canonical trajectories without ICU dynamics."""
    selected_cases: list[dict] = []
    sequence_frames: list[pd.DataFrame] = []
    for row in selected_rows:
        trajectories = _load_trajectories(Path(row["output_dir"]) / "trajectories.csv")
        success_column = "success" if "success" in trajectories else "survived"
        grouped = trajectories.sort_values(["episode_id", "t"]).groupby("episode_id", sort=False)
        episode_rows = []
        for episode_id, frame in grouped:
            episode_rows.append(
                {
                    "method_id": row["method_id"],
                    "episode_id": int(episode_id),
                    "return": float(frame["reward"].sum()),
                    "success": bool(frame[success_column].iloc[-1]),
                    "length": int(len(frame)),
                }
            )
        candidates = pd.DataFrame(episode_rows).sort_values(["return", "success"], ascending=[False, False]).head(args.top_k)
        selected_cases.extend(candidates.to_dict("records"))
        selected_ids = set(candidates["episode_id"].tolist())
        sequence = trajectories[trajectories["episode_id"].isin(selected_ids) & (trajectories["t"] < args.horizon)].copy()
        sequence["method_id"] = row["method_id"]
        sequence_frames.append(sequence)
    cases = pd.DataFrame(selected_cases)
    sequences = pd.concat(sequence_frames, ignore_index=True) if sequence_frames else pd.DataFrame()
    cases_path = output_dir / "generic_selected_cases.csv"
    sequences_path = output_dir / "generic_case_sequences.csv"
    markdown_path = output_dir / "generic_selected_cases.md"
    cases.to_csv(cases_path, index=False)
    sequences.to_csv(sequences_path, index=False)
    lines = ["# Qualitative trajectories", ""]
    for row in cases.itertuples(index=False):
        lines.append(
            f"- `{row.method_id}` episode {row.episode_id}: "
            f"return={getattr(row, 'return'):.3f}, success={int(row.success)}, length={row.length}"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"selected_path": cases_path, "sequences_path": sequences_path, "markdown_path": markdown_path}


def run_selection(args: argparse.Namespace) -> dict[str, Path]:
    manifest_rows = _load_manifest(args.manifest)
    selected_rows = _filter_manifest_rows(manifest_rows, dataset_tag=args.dataset_tag, methods=args.methods)
    output_dir = _resolve_output_dir(args.manifest, dataset_tag=args.dataset_tag, output_dir=args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    first_config = Path(selected_rows[0]["output_dir"]) / "config.json"
    if first_config.exists():
        with first_config.open("r", encoding="utf-8") as handle:
            if json.load(handle).get("env_id", "icu_sepsis") != "icu_sepsis":
                return _run_generic_selection(args, selected_rows, output_dir)

    dynamics = _load_dynamics(args.dynamics_path)
    centers = np.asarray(dynamics["state_cluster_centers"], dtype=np.float64)
    sofa_scores = np.asarray(dynamics["sofa_scores"], dtype=np.float64)
    expert_policy = np.asarray(dynamics["expert_policy"], dtype=np.float64)
    optimal_policy = _compute_optimal_policy(
        np.asarray(dynamics["tx_mat"], dtype=np.float64),
        np.asarray(dynamics["r_mat"], dtype=np.float64),
        gamma=1.0,
    )

    sequences_by_method: dict[str, list[EpisodeSequence]] = {}
    metadata_rows: list[dict] = []
    for row in selected_rows:
        method_id = row["method_id"]
        trajectories_path = Path(row["output_dir"]) / "trajectories.csv"
        trajectories = _load_trajectories(trajectories_path)
        sequences = build_episode_sequences(trajectories, horizon=args.horizon)
        if not sequences:
            raise ValueError(f"No episodes with horizon>={args.horizon} found for {method_id}: {trajectories_path}")
        sequences_by_method[method_id] = sequences
        metadata_rows.append(
            {
                "method_id": method_id,
                "method_label": DISPLAY_NAMES.get(method_id, method_id),
                "dataset_tag": row["dataset_tag"],
                "output_dir": row["output_dir"],
                "num_supported_episodes": len(sequences),
            }
        )

    candidates, sequences = summarize_initial_states(
        sequences_by_method=sequences_by_method,
        methods=args.methods,
        horizon=args.horizon,
        min_support=args.min_support,
        length_bucket_size=args.length_bucket_size,
        stratify_by_outcome=args.stratify_by_outcome,
        centers=centers,
        sofa_scores=sofa_scores,
        expert_policy=expert_policy,
        optimal_policy=optimal_policy,
    )
    if candidates.empty:
        raise ValueError(
            "No candidate initial states remained after filtering. "
            f"Try lowering --min-support (currently {args.min_support})."
        )

    selected = _select_case_mix(candidates, top_k=args.top_k)
    selected_case_keys = set(selected["candidate_key"].tolist())
    selected_sequences = sequences[sequences["candidate_key"].isin(selected_case_keys)].copy()
    case_rank_map = dict(zip(selected["candidate_key"], selected["case_rank"]))
    if not selected_sequences.empty:
        selected_sequences["case_rank"] = selected_sequences["candidate_key"].map(case_rank_map)
        selected_sequences = selected_sequences.sort_values(["case_rank", "method_label"]).reset_index(drop=True)

    candidate_path = output_dir / "initial_state_candidates.csv"
    selected_path = output_dir / "selected_cases.csv"
    sequences_path = output_dir / "case_sequences.csv"
    vectors_path = output_dir / "case_initial_state_vectors.csv"
    markdown_path = output_dir / "selected_cases.md"
    markdown_no_optimal_path = output_dir / "selected_cases_sem_acao_otima.md"
    review_payload_path = output_dir / "review_payload.json"
    metadata_path = output_dir / "selection_metadata.json"

    candidates.to_csv(candidate_path, index=False)
    selected.to_csv(selected_path, index=False)
    selected_sequences.to_csv(sequences_path, index=False)
    vector_rows = []
    interpretation_rows = []
    for _, case in selected.iterrows():
        initial_state = int(case["initial_state"])
        row = {
            "case_rank": int(case["case_rank"]),
            "selection_bucket": case["selection_bucket"],
            "initial_state": initial_state,
            "initial_sofa": float(case["initial_sofa"]),
        }
        center = centers[initial_state]
        for feature_name, value in zip(FEATURE_NAMES, center):
            row[feature_name] = float(value)
            raw_status, interpretation = _feature_interpretation(feature_name, float(value))
            interpretation_rows.append(
                {
                    "case_rank": int(case["case_rank"]),
                    "selection_bucket": case["selection_bucket"],
                    "initial_state": initial_state,
                    "feature_name": feature_name,
                    "feature_type": _feature_type(feature_name),
                    "normalized_value": float(value),
                    "raw_value_status": raw_status,
                    "best_available_interpretation": interpretation,
                }
            )
        vector_rows.append(row)
    pd.DataFrame(vector_rows).to_csv(vectors_path, index=False)
    interpretations_path = output_dir / "case_initial_state_feature_interpretation.csv"
    pd.DataFrame(interpretation_rows).to_csv(interpretations_path, index=False)
    review_payload_path.write_text(
        json.dumps(
            _build_review_payload(selected, selected_sequences, args.methods),
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    _write_markdown(
        markdown_path,
        selected,
        selected_sequences,
        methods=args.methods,
        centers=centers,
        include_optimal_actions=True,
    )
    _write_markdown(
        markdown_no_optimal_path,
        selected,
        selected_sequences,
        methods=args.methods,
        centers=centers,
        include_optimal_actions=False,
    )
    metadata_path.write_text(
        json.dumps(
            {
                "manifest": str(args.manifest),
                "dataset_tag": args.dataset_tag,
                "methods": args.methods,
                "horizon": args.horizon,
                "min_support": args.min_support,
                "length_bucket_size": args.length_bucket_size,
                "stratify_by_outcome": args.stratify_by_outcome,
                "top_k": args.top_k,
                "dynamics_path": str(args.dynamics_path),
                "loaded_runs": metadata_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "candidate_path": candidate_path,
        "selected_path": selected_path,
        "sequences_path": sequences_path,
        "vectors_path": vectors_path,
        "interpretations_path": interpretations_path,
        "review_payload_path": review_payload_path,
        "markdown_path": markdown_path,
        "markdown_no_optimal_path": markdown_no_optimal_path,
        "metadata_path": metadata_path,
    }


def main() -> None:
    args = _parse_args()
    paths = run_selection(args)
    print("Qualitative-case selection complete.")
    for label, path in paths.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
