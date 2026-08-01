import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd

STRATEGY_LABELS = {
    'happo': 'OmniSearch RL',
    'ant_colony': 'ACO',
    'lawnmower': 'Lawnmower',
    'random_walk': 'Random Walk',
    'random_action': 'Random Action',
    'highest_confidence': 'Highest Confidence',
}

def _scenario_number(scenario, key):
    value = scenario.get(key, float('nan'))
    return float(value) if isinstance(value, (int, float, np.floating)) and np.isfinite(value) else float('nan')

def _survivor_count_value(count):
    return int(count)

def _run_identity(
    path,
    payload,
    file_index,
    run_identities=None,
    json_files=None,
    strategies=None,
):
    if (
        run_identities is not None
        and json_files is not None
        and len(run_identities) == len(json_files)
    ):
        strategy, count = run_identities[file_index]
        return _strategy_tag(strategy), int(count)

    metadata = payload.get('metadata', {})
    payload_strategy = payload.get('strategy') or payload.get('approach')
    if payload_strategy is None and isinstance(metadata, dict):
        payload_strategy = metadata.get('strategy') or metadata.get('approach')
    stem = path.stem.lower()
    configured_strategies = [
        _strategy_tag(strategy)
        for strategy in (strategies if strategies is not None else STRATEGY_LABELS)
    ]
    strategy = _strategy_tag(payload_strategy) if payload_strategy else next(
        (name for name in configured_strategies if stem.startswith(f'{name}_')),
        'unknown',
    )

    scenario = _payload_scenario(payload)
    active_min = _scenario_number(scenario, 'active_survivors_min')
    active_max = _scenario_number(scenario, 'active_survivors_max')
    if np.isfinite(active_min) and np.isfinite(active_max) and active_min == active_max:
        count = int(active_min)
    else:
        match = re.search(r'_survivors_(\d+)_seeds_', stem)
        count = int(match.group(1)) if match else float('nan')
    return strategy, count

def _resolve_json_file(path, project_root):
    path = Path(path).expanduser()
    candidates = [path]
    if not path.is_absolute():
        candidates.extend([project_root / path, project_root / 'notebooks' / path])
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()

def _run_diagnostic_job(strategy, cmd, log_path, project_root, dropout=0.0):
    with log_path.open('w') as log:
        subprocess.run(
            cmd,
            cwd=project_root,
            check=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    return strategy, dropout

def _diagnostic_prefix(strategy, checkpoint, project_root, happo_script, baseline_script):
    strategy_tag = _strategy_tag(strategy)
    if strategy_tag == 'happo':
        return [
            sys.executable,
            str(project_root / happo_script),
            '--checkpoint-dir', str(checkpoint),
        ]
    return [
        sys.executable,
        str(project_root / baseline_script),
        '--strategy', strategy_tag,
        '--happo-checkpoint', str(checkpoint),
    ]

def _resolve_checkpoint_for_count(count, ckpt, project_root, results_root, model_label):
    count = int(count)
    if count not in ckpt:
        raise KeyError(f'No checkpoint configured for survivor count {count}')
    configured = ckpt[count]
    if configured is not None:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            path = project_root / path
        if not path.is_dir():
            raise FileNotFoundError(f'Checkpoint directory not found for n={count}: {path}')
        return path.resolve()

    run_label = f'happo_{model_label}_{count}'
    candidates = sorted((results_root / run_label).glob('seed-*/models'), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(
            f'No checkpoint found under {results_root / run_label}. '
            f'Set CHECKPOINTS_BY_SURVIVOR_COUNT[{count}] explicitly.'
        )
    return candidates[-1].resolve()

def _resolve_checkpoint(ckpt, project_root, results_root, model_label):
    if ckpt is not None:
        path = Path(ckpt).expanduser()
        if not path.is_absolute():
            path = project_root / path
        if not path.is_dir():
            raise FileNotFoundError(f'Checkpoint directory not found: {path}')
        return path.resolve()

    candidates = sorted((results_root / model_label).glob('seed-*/models'), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(
            f'No checkpoint found under {results_root / model_label}. Set CKPT explicitly.'
        )
    return candidates[-1].resolve()

def _strategy_tag(strategy):
    normalized = str(strategy).strip().lower().replace('-', '_').replace(' ', '_')
    for tag in STRATEGY_LABELS:
        if normalized == tag or normalized.startswith(f'{tag}_'):
            return tag
    return normalized

def _summary_value(summary, *keys, default=float('nan')):
    for key in keys:
        if key in summary:
            return summary[key]
    return default

def _dropout_probability(dropout):
    value = float(dropout)
    return value / 100.0 if value > 1.0 else value

def _dropout_cli_value(dropout):
    return f'{_dropout_probability(dropout):g}'


def _dropout_tag(dropout):
    value = _dropout_probability(dropout)
    return f'{value:.1f}'.replace('.', 'p')

def _diagnostic_output_paths(
    strategy,
    sweep_value=None,
    *,
    model_label,
    output_dir,
    seeds,
    comms_dropout=None,
    survivor_count=None,
):
    """Return stable output paths for dropout and survivor-load sweeps.

    Existing notebooks pass their sweep value positionally. Values in [0, 1]
    are communication-dropout probabilities; larger values are survivor counts.
    New callers can avoid that shorthand with the explicit keyword arguments.
    """
    if sweep_value is not None:
        if comms_dropout is not None or survivor_count is not None:
            raise ValueError(
                'sweep_value cannot be combined with comms_dropout or survivor_count'
            )
        numeric_value = float(sweep_value)
        if 0.0 <= numeric_value <= 1.0:
            comms_dropout = numeric_value
        elif numeric_value > 1.0 and numeric_value.is_integer():
            survivor_count = int(numeric_value)
        else:
            raise ValueError(
                'positional sweep_value must be a dropout in [0, 1] or a '
                'positive integer survivor count'
            )

    if comms_dropout is not None and survivor_count is not None:
        raise ValueError('specify either comms_dropout or survivor_count, not both')

    strategy_tag = str(strategy).replace('-', '_')
    if survivor_count is not None:
        survivor_count = int(survivor_count)
        if survivor_count < 1:
            raise ValueError('survivor_count must be at least 1')
        stem = (
            f'{strategy_tag}_{model_label}_{survivor_count}_'
            f'seeds_{seeds[0]}_{seeds[1]}'
        )
    else:
        tag = _dropout_tag(0.0 if comms_dropout is None else comms_dropout)
        stem = (
            f'{strategy_tag}_{model_label}_dropout{tag}_'
            f'seeds_{seeds[0]}_{seeds[1]}'
        )
    return output_dir / f'{stem}.json', output_dir / f'{stem}.png'


def _payload_scenario(payload):
    scenario = payload.get("scenario")
    if isinstance(scenario, dict) and scenario:
        return scenario

    scenario_kwargs = payload.get("scenario_kwargs")
    if isinstance(scenario_kwargs, dict) and scenario_kwargs:
        return scenario_kwargs

    metadata = payload.get("metadata", {})
    scenario_kwargs = (
        metadata.get("scenario_kwargs")
        if isinstance(metadata, dict)
        else None
    )
    return scenario_kwargs if isinstance(scenario_kwargs, dict) else {}

def _strategy_from_payload(path, payload):
    # Prefer the explicit summary strategy.
    summary = payload.get("summary", {})
    strategy = summary.get("strategy") if isinstance(summary, dict) else None
    if isinstance(strategy, str) and strategy:
        return strategy

    # Then inspect metadata, supporting both dictionary and string formats.
    metadata = payload.get("metadata", {})
    strategy = metadata.get("strategy") if isinstance(metadata, dict) else None

    if isinstance(strategy, dict):
        strategy = strategy.get("name") or strategy.get("label")

    if isinstance(strategy, str) and strategy:
        return strategy

    # HAPPO JSON files often have no strategy metadata, so infer from filename.
    normalized_stem = path.stem.lower().replace("-", "_")
    for candidate in STRATEGY_LABELS:
        if candidate in normalized_stem:
            return candidate

    return path.stem

def _label_for_identity(strategy, count, path):
    strategy_label = STRATEGY_LABELS.get(strategy, strategy.replace('_', ' ').title())
    if isinstance(count, (int, float, np.floating)) and np.isfinite(count):
        return f'{strategy_label} | n={int(count)}'
    return f'{strategy_label} | {path.stem}'

def _label_from_payload(path, payload, *, include_dropout=False):
    strategy = _strategy_from_payload(path, payload)
    strategy_tag = _strategy_tag(strategy)

    label = STRATEGY_LABELS.get(
        strategy_tag,
        str(strategy).replace("_", " ").title(),
    )

    if not include_dropout:
        return label

    scenario = _payload_scenario(payload)
    dropout = scenario.get("comms_dropout")
    mode = scenario.get("comms_dropout_mode")

    if dropout is None:
        return label

    return f'{label} | dropout={float(dropout):g}, mode={mode or "?"}'


def _success_rate(summary):
    rate = _summary_value(summary, 'full_confirm_success_rate', 'success_rate')
    if pd.isna(rate):
        percent = _summary_value(summary, 'full_confirm_success_percent')
        rate = percent / 100.0 if not pd.isna(percent) else float('nan')
    return rate


def _success_std(summary, rate):
    if pd.isna(rate):
        return float('nan')
    episodes = _summary_value(summary, 'episodes')
    if pd.isna(episodes) or episodes <= 0:
        return float('nan')
    return float(np.sqrt(max(rate * (1.0 - rate), 0.0)))


def _finite(values):
    return [float(value) for value in values if isinstance(value, (int, float, np.floating)) and np.isfinite(value)]


def _mean_std(values):
    values = _finite(values)
    if not values:
        return float('nan'), float('nan')
    return float(np.mean(values)), float(np.std(values))


def _episode_steps(row, scenario, nsteps_per_episode=900):
    for key in ('episode_steps', 'max_steps'):
        value = row.get(key)
        if isinstance(value, (int, float)) and value > 0:
            return int(value)
    value = scenario.get('max_steps')
    if isinstance(value, (int, float)) and value > 0:
        return int(value)
    return int(nsteps_per_episode)


def _active_survivor_indices(row):
    explicit = row.get('active_survivor_indices')
    if isinstance(explicit, list):
        return [int(idx) for idx in explicit]
    mask = row.get('active_survivor_mask')
    if isinstance(mask, list):
        return [idx for idx, active in enumerate(mask) if bool(active)]
    steps = row.get('first_scout_steps') or row.get('first_confirm_steps') or []
    count = row.get('active_survivors', row.get('survivors', len(steps)))
    try:
        count = int(count)
    except (TypeError, ValueError):
        count = len(steps)
    return list(range(min(count, len(steps))))


def _auc_from_first_steps(row, scenario, key, nsteps_per_episode=900):
    first_steps = row.get(key)
    if not isinstance(first_steps, list):
        return float('nan')
    steps = max(_episode_steps(row, scenario, nsteps_per_episode), 1)
    indices = _active_survivor_indices(row)
    if not indices:
        return 1.0
    total = 0.0
    for idx in indices:
        if idx >= len(first_steps):
            continue
        first_step = first_steps[idx]
        if first_step is None:
            continue
        try:
            first_step = float(first_step)
        except (TypeError, ValueError):
            continue
        if first_step <= 0:
            total += 1.0
        elif first_step <= steps:
            total += (steps - first_step + 1.0) / steps
    return float(total / max(len(indices), 1))


def _row_values(rows, *keys):
    values = []
    for row in rows:
        for key in keys:
            value = row.get(key)
            if isinstance(value, (int, float, np.floating)) and np.isfinite(value):
                values.append(float(value))
                break
    return values


def _summary_or_rows(summary, rows, mean_key, std_key, *row_keys, fallback=None):
    mean = _summary_value(summary, mean_key)
    std = _summary_value(summary, std_key)
    if not pd.isna(mean):
        return float(mean), float(std) if not pd.isna(std) else float('nan')
    values = _row_values(rows, *row_keys)
    if not values and fallback is not None:
        values = [fallback(row) for row in rows]
    return _mean_std(values)


def _ugv_cost_per_confirm_stats(rows):
    totals = []
    confirmed = []
    for row in rows:
        total = row.get('ugv_travel_cost_total')
        count = row.get('confirmed')
        if not isinstance(total, (int, float, np.floating)) or not np.isfinite(total):
            continue
        if not isinstance(count, (int, float, np.floating)) or not np.isfinite(count) or count < 0:
            continue
        totals.append(float(total))
        confirmed.append(float(count))
    if not totals or sum(confirmed) <= 0:
        return float('nan'), float('nan')
    aggregate_mean = float(sum(totals) / sum(confirmed))
    ratios = np.asarray(
        [total / count for total, count in zip(totals, confirmed) if count > 0],
        dtype=float,
    )
    weights = np.asarray([count for count in confirmed if count > 0], dtype=float)
    weighted_variance = np.average((ratios - aggregate_mean) ** 2, weights=weights)
    return aggregate_mean, float(np.sqrt(weighted_variance))


def _uav_confidence_efficiency_stats(rows, scenario):
    n_drones = int(scenario.get('n_drones', scenario.get('obs_schema_n_drones', 0)) or 0)
    distances_km = []
    confidence_area_by_uav = []
    for row in rows:
        path_m = row.get('uav_path_length_m')
        confidence = row.get('final_confidence_mean')
        width_m = row.get('map_width_m')
        height_m = row.get('map_height_m')
        values = (path_m, confidence, width_m, height_m)
        if not all(isinstance(value, (int, float, np.floating)) for value in values):
            continue
        if not all(np.isfinite(value) for value in values) or n_drones <= 0:
            continue
        map_area_km2 = float(width_m) * float(height_m) / 1_000_000.0
        confidence_output = n_drones * map_area_km2 * max(float(confidence), 0.0)
        distances_km.append(float(path_m) / 1000.0)
        confidence_area_by_uav.append(confidence_output)
    if not distances_km or sum(confidence_area_by_uav) <= 0:
        return float('nan'), float('nan')
    aggregate_mean = float(sum(distances_km) / sum(confidence_area_by_uav))
    ratios = np.asarray(
        [
            distance / output
            for distance, output in zip(distances_km, confidence_area_by_uav)
            if output > 0
        ],
        dtype=float,
    )
    weights = np.asarray([output for output in confidence_area_by_uav if output > 0], dtype=float)
    weighted_variance = np.average((ratios - aggregate_mean) ** 2, weights=weights)
    return aggregate_mean, float(np.sqrt(weighted_variance))


def _threshold_time(summary, threshold_key):
    container = summary.get('time_to_confirm_s', {})
    stats = container.get(threshold_key, {}) if isinstance(container, dict) else {}
    if not isinstance(stats, dict):
        stats = {}
    return {
        'mean': float(stats.get('mean_s', float('nan'))),
        'std': float(stats.get('std_s', float('nan'))),
        'reached_fraction': float(stats.get('reached_fraction', float('nan'))),
        'reached_count': float(stats.get('reached_count', float('nan'))),
    }


def _latency_stats(summary):
    fast = summary.get('fast_metrics', {})
    latency = fast.get('scout_to_confirm_latency_s') if isinstance(fast, dict) else None
    if isinstance(latency, dict):
        mean = latency.get('mean', float('nan'))
        std = latency.get('std', float('nan'))
    else:
        mean = _summary_value(summary, 'mean_scout_to_confirm_latency_s')
        std = _summary_value(summary, 'std_scout_to_confirm_latency_s')
    return float(mean), float(std)
