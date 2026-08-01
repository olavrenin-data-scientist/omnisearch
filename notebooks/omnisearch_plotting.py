import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import textwrap
from matplotlib.colors import LinearSegmentedColormap

PAPER_BG = '#f2ebd3ff'
#PAPER_BG = 'white'
GRID_COLOR = '#cfc7ad'
TEXT_GREEN = '#23562f'
TEXT_DARK = '#2f2a24'
#TEXT_GREEN = '#002676'
MISSING_COLOR = '#d8d1bd'
SET_TITLE = False

STRATEGY_LABELS = {
    'happo': 'OmniSearch RL',
    'ant_colony': 'ACO',
    'lawnmower': 'Lawnmower',
    'random_walk': 'Random Walk',
    'random_action': 'Random Action',
    'highest_confidence': 'Highest Confidence',
}

TERRAIN_LABELS = {
    'malibu': 'Malibu Creek State Park',
    'aubern': 'Auburn SRA',
    'topanga': 'Topanga State Park',
    'san_marcos': 'San Marcos Foothills',
}

AUC_METRIC_SPECS = [
    ('success', 'Success', '#8f2418', 's'),
    ('confirm_auc', 'Confirm AUC', '#c94a27', 's'),
    ('scout_auc', 'Scout AUC', '#e09b4f', 'o'),
    ('confidence_auc', 'Confidence AUC', '#1f5a35', '^'),
    ('coverage_auc', 'Coverage AUC', '#5f812f', 'v'),
]
TIME_METRIC_SPECS = [
    ('confirm_time_100', '100% Confirm', '#8f2418'),
    ('confirm_time_080', '80% Confirm', '#c94a27'),
    ('confirm_time_050', '50% Confirm', '#1f5a35'),
    ('scout_to_confirm_latency', 'Scout-to-Confirm', '#5f812f'),
]
tradeoff_colors = ['#9d2b1f', '#cf4a27', '#1f5a35', '#608633', '#e09b4f']

plt.rcParams.update({
    'figure.facecolor': PAPER_BG,
    'axes.facecolor': PAPER_BG,
    'axes.edgecolor': '#f2ebd3ff',
#    'axes.edgecolor': '#9d9279',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.titleweight': 'bold',
    'axes.titlesize': 24,
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'text.color': TEXT_DARK,
    'axes.labelcolor': TEXT_DARK,
    'xtick.color': TEXT_DARK,
    'ytick.color': TEXT_DARK,
    'legend.frameon': True,
    'font.family': 'serif',
})

cmap = LinearSegmentedColormap.from_list(
    'paper_confirm_auc',
    ['#f0d9a6', '#e09b4f', '#9ba04d', '#23562f'],
)
cmap.set_bad(MISSING_COLOR)

def _available_specs(specs, df, *, mean_suffix='_mean'):
    keep = []
    for spec in specs:
        prefix = spec[0]
        col = f'{prefix}{mean_suffix}'
        if col in df.columns and pd.to_numeric(df[col], errors='coerce').notna().any():
            keep.append(spec)
    return keep

def _style_axis(ax):
    ax.grid(axis='y', color=GRID_COLOR, alpha=0.55, linewidth=0.9)
    ax.grid(axis='x', color=GRID_COLOR, alpha=0.30, linewidth=0.9)
    ax.spines['left'].set_color('#9d9279')
    ax.spines['bottom'].set_color('#9d9279')

def _add_bar_labels(ax, bars, std, *, offset_rank=0, suffix='', digits=2, show_error=True):
    for bar, err in zip(bars, std):
        height = bar.get_height()
        if not np.isfinite(height):
            continue
        err = 0.0 if not np.isfinite(err) else float(err)
        label_err = err if show_error else 0.0
        label_text = f'{height:.{digits}f}{suffix}'
        if show_error:
            label_text = f'{height:.{digits}f}\u00b1{err:.{digits}f}{suffix}'
        ax.annotate(
            label_text,
            xy=(bar.get_x() + bar.get_width() / 2.0, height + label_err),
            xytext=(0, 22 + 5 * (offset_rank % 2)),
            textcoords='offset points',
            ha='center',
            va='bottom',
            fontsize=16.5,
            fontweight='bold',
            rotation=90,
            linespacing=0.9,
            color=TEXT_DARK,
            clip_on=False,
        )

def _plot_metrics_bars(ax, df, specs, *, title, ylabel, mean_suffix='_mean', std_suffix='_std', y_limit=None, x_labels='label', label_suffix='', label_digits=2):
    specs = _available_specs(specs, df, mean_suffix=mean_suffix)
    if not specs:
        ax.set_axis_off()
        ax.set_title(f'{title} not available', color=TEXT_GREEN, pad=12)
        return
    positions = np.arange(len(df))
    width = min(0.155, 0.76 / max(len(specs), 1))
    offset_units = np.arange(len(specs), dtype=float) - (len(specs) - 1) / 2.0
    series = []
    ymax = 1.0
    for prefix, label, color, *_rest in specs:
        N=df["episodes"].tolist()
        means = pd.to_numeric(df[f'{prefix}{mean_suffix}'], errors='coerce').to_numpy(dtype=float)
        stds = pd.to_numeric(df[f'{prefix}{std_suffix}'], errors='coerce').to_numpy(dtype=float)/np.sqrt(N)*1.96
        plot_mean = np.where(np.isfinite(means), means, 0.0)
        #show_error = prefix != 'success'
        show_error = True
        plot_std = np.where(np.isfinite(stds) & show_error, stds, 0.0)
        if len(plot_mean):
            ymax = max(ymax, float(np.nanmax(plot_mean + plot_std)) * 1.45)
        series.append((prefix, label, color, means, stds, plot_mean, plot_std, show_error))
    if y_limit is None:
        ax.set_ylim(0.0, ymax)
    else:
        ax.set_ylim(*y_limit)
    for offset, (_prefix, label, color, means, stds, plot_mean, plot_std, show_error) in enumerate(series):
        bar_x = positions + offset_units[offset] * width
        bars = ax.bar(
            bar_x,
            plot_mean,
            width=width,
            yerr=plot_std,
            capsize=3 if show_error else 0,
            color=color,
            alpha=0.98,
            label=label,
            edgecolor=PAPER_BG,
            linewidth=0.8,
            error_kw={'elinewidth': 1.0, 'alpha': 0.65, 'ecolor': TEXT_DARK},
        )
        _add_bar_labels(ax, bars, stds, offset_rank=offset, suffix=label_suffix, digits=label_digits, show_error=show_error)
    if x_labels == 'strategy':
        labels = [STRATEGY_LABELS[str_strategy] for str_strategy in df[x_labels].tolist()]
    elif x_labels == 'terrain':
        labels = [TERRAIN_LABELS[str_terrain] for str_terrain in df[x_labels].tolist()]
    else:
        labels = df[x_labels].str.replace(r' \| .*$', '', regex=True).tolist()
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=0, ha='center', fontweight='bold', fontsize=16)
    if SET_TITLE == True:
        ax.set_title(title, color=TEXT_GREEN, pad=12)
    ax.set_ylabel(ylabel)
    _style_axis(ax)

    legend = ax.legend(
        title="Mean with 95% CI",
        title_fontsize=14,
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        fontsize=14,
    )
    legend.get_frame().set_facecolor(PAPER_BG)
    legend.get_frame().set_edgecolor('#c8bea4')
    legend.get_frame().set_alpha(0.90)

def _plot_single_metric_bars(
    ax,
    df,
    *,
    mean_column,
    std_column,
    scale,
    title,
    ylabel,
    color,
    label_digits,
):
    N = df["episodes"].tolist()

    means = pd.to_numeric(df[mean_column], errors='coerce').to_numpy(dtype=float) * scale
    stds = pd.to_numeric(df[std_column], errors='coerce').to_numpy(dtype=float) * scale / np.sqrt(N)*1.96
    positions = np.arange(len(df))
    plot_means = np.where(np.isfinite(means), means, 0.0)
    plot_stds = np.where(np.isfinite(stds), stds, 0.0)
    upper = plot_means + plot_stds
    ymax = float(np.max(upper)) * 1.35 if len(upper) and np.max(upper) > 0 else 1.0
    bars = ax.bar(
        positions,
        plot_means,
        width=0.62,
        yerr=plot_stds,
        capsize=5,
        color=color,
        edgecolor=PAPER_BG,
        linewidth=0.9,
        error_kw={'elinewidth': 1.2, 'alpha': 0.70, 'ecolor': TEXT_DARK},
    )
    for bar, err in zip(bars, stds):
        height = bar.get_height()
        if not np.isfinite(height):
            continue
        err = float(err) if np.isfinite(err) else 0.0
        ax.annotate(
            f'{height:.{label_digits}f}±{err:.{label_digits}f}',
            xy=(bar.get_x() + bar.get_width() / 2.0, height + err),
            xytext=(0, 8),
            textcoords='offset points',
            ha='center',
            va='bottom',
            fontsize=13.5,
            fontweight='bold',
            color=TEXT_DARK,
            clip_on=False,
        )
    labels = df['label'].str.replace(r' \| .*$', '', regex=True).tolist()
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontweight='bold', fontsize=15)
    ax.set_ylim(0.0, ymax)
    if SET_TITLE == True:
        ax.set_title(title, color=TEXT_GREEN, pad=12)
    ax.set_ylabel(ylabel)
    _style_axis(ax)

def _plot_confirm_auc_curve(ax, df, title, x_title, x_label, strategy_labels=[],print_percentage=['happo'],y_limit=None):
    markers = ['o', 's', '^', 'v']
    groups = df.groupby('strategy', sort=False) if 'strategy' in df.columns else [('results', df)]
    plotted = []
    all_x = []
    all_lower = []
    all_upper = []
    strategy_colors = {
           'lawnmower': '#5f812f',
           'ant_colony': '#23562f',
           'happo': '#9d2b1f',
       }
    fallback_colors = [ '#9d2b1f','#23562f','#5f812f', '#e09b4f']
    if x_label == 'terrain':
        terrain_order = list(df['terrain'].unique().tolist())
        terrain_positions = np.arange(len(terrain_order), dtype=float)
    for group_index, (strategy, group) in enumerate(groups):
        if x_label == 'terrain':
            x = terrain_positions
        else:
            group = group.sort_values(x_label)
            x = pd.to_numeric(group[x_label], errors='coerce').to_numpy(dtype=float)
        mean = pd.to_numeric(group['confirm_auc_mean'], errors='coerce').to_numpy(dtype=float)
        std = pd.to_numeric(group['confirm_auc_std'], errors='coerce').to_numpy(dtype=float)/np.sqrt(100)*1.96
        success = pd.to_numeric(group['success_mean'], errors='coerce').to_numpy(dtype=float)

        valid = np.isfinite(x) & np.isfinite(mean)
        x = x[valid]
        mean = mean[valid]
        std = np.where(np.isfinite(std[valid]), std[valid], 0.0)
        if not len(x):
            continue

        lower = np.clip(mean - std, 0.0, 1.0)
        upper = np.clip(mean + std, 0.0, 1.0)
        color = strategy_colors.get(
                    strategy,
                    fallback_colors[group_index % len(fallback_colors)],
                )
        if strategy_labels==[]:
            label = STRATEGY_LABELS[strategy]
        else: label=strategy_labels[group_index]
        ax.fill_between(x, lower, upper, color=color, alpha=0.12, linewidth=0)
        ax.plot(
            x,
            mean,
            color=color,
            marker=markers[group_index % len(markers)],
            markersize=8,
            linewidth=3.0,
            label=label,
        )
        if strategy in print_percentage:
            for xi, mean, success_rate in zip(x, mean, success):
                if xi == 1.0 and x_label=='dropout_x':
                    continue
                else:
                    ax.annotate(
                        f'{success_rate:.0%}',
                        xy=(xi, mean),
                        xytext=(0, 9),
                        textcoords='offset points',
                        ha='center',
                        va='bottom',
                        color=color,
                        fontsize=20,
                        fontweight='bold',
                                )
        plotted.append(label)
        all_x.append(x)
        all_lower.append(lower)
        all_upper.append(upper)

    if not plotted:
        ax.set_axis_off()
        ax.set_title('Confirm AUC vs Dropout not available', color=TEXT_GREEN, pad=12)
        return

    x = np.concatenate(all_x)
    lower = np.concatenate(all_lower)
    upper = np.concatenate(all_upper)
    x_margin = max((float(np.nanmax(x)) - float(np.nanmin(x))) * 0.06, 0.01)
    y_min = max(float(np.nanmin(lower)) - 0.08, 0.0)
    y_max = min(max(float(np.nanmax(upper)) + 0.16, y_min + 0.2), 1.10)
    ax.set_xlim(float(np.nanmin(x)) - x_margin, float(np.nanmax(x)) + x_margin)
    if y_limit is None:
        ax.set_ylim(y_min, y_max)
    else:
        ax.set_ylim(*y_limit)
    ax.set_xticks(sorted(set(x.tolist())))
    if x_label == "terrain":
        terrain_labels = [TERRAIN_LABELS[ter] for ter in terrain_order]
        ax.set_xticklabels(_wrapped_labels(terrain_labels), fontweight='bold', fontsize=18)
    if SET_TITLE==True:
        ax.set_title(title, color=TEXT_GREEN, pad=12)
    ax.set_xlabel(x_title)
    ax.set_ylabel('Confirm AUC')
    _style_axis(ax)
    if len(plotted) > 0:
        legend = ax.legend(title ='Mean \u00b1 95% CI', title_fontsize=20, loc='best', fontsize=20)
        legend.get_frame().set_facecolor(PAPER_BG)
        legend.get_frame().set_edgecolor('#c8bea4')
        legend.get_frame().set_alpha(0.90)

def _wrapped_labels(labels, width=16):
    return ['\n'.join(textwrap.wrap(str(label), width=width)) or str(label) for label in labels]

def _plot_efficiency_tradeoff(
    ax,
    df,
    *,
    x_mean_column,
    x_std_column,
    y_mean_column,
    y_std_column,
    success_column,
    success_label,
    title,
    xlabel,
    ylabel,
    label_offsets,
):
    tradeoff_df = df.dropna(subset=[x_mean_column, y_mean_column]).reset_index(drop=True)
    for index, row in tradeoff_df.iterrows():
        label = str(row['label']).split(' | ', 1)[0]
        x = float(row[x_mean_column])
        y = float(row[y_mean_column])
        xerr = float(row[x_std_column])/np.sqrt(100)*1.96 if np.isfinite(row[x_std_column]) else 0.0
        yerr = float(row[y_std_column])/np.sqrt(100)*1.96 if np.isfinite(row[y_std_column]) else 0.0
        print(y, float(row[y_std_column]),yerr, y+yerr, y-yerr)
        success = float(row[success_column]) if np.isfinite(row[success_column]) else 0.0
        color = tradeoff_colors[index % len(tradeoff_colors)]
        ax.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            fmt='none',
            ecolor=color,
            elinewidth=2.0,
            capsize=5,
            alpha=0.65,
            zorder=2,
        )
        ax.scatter(
            x,
            y,
            s=52.0 + 124.0 * np.clip(success, 0.0, 1.0),
            color=color,
            edgecolor=PAPER_BG,
            linewidth=2.0,
            zorder=3,
        )
        offset = label_offsets.get(label, (12, 14))
        ax.annotate(
            f'{label}\n{success:.0%} {success_label}',
            xy=(x, y),
            xytext=offset,
            textcoords='offset points',
            ha='left',
            va='bottom' if offset[1] >= 0 else 'top',
            fontsize=22,
            fontweight='bold',
            color=color,
            zorder=4,
        )

    if not tradeoff_df.empty:
        x_values = tradeoff_df[x_mean_column].to_numpy(dtype=float)
        x_errors = tradeoff_df[x_std_column].fillna(0.0).to_numpy(dtype=float)/np.sqrt(100)*1.96
        y_values = tradeoff_df[y_mean_column].to_numpy(dtype=float)
        y_errors = tradeoff_df[y_std_column].fillna(0.0).to_numpy(dtype=float)/np.sqrt(100)*1.96

        x_low, x_high = np.min(x_values - x_errors), np.max(x_values + x_errors)
        y_low, y_high = np.min(y_values - y_errors), np.max(y_values + y_errors)
        x_pad = max((x_high - x_low) * 0.16, 0.025)
        y_pad = max((y_high - y_low) * 0.16, 0.20)
        ax.set_xlim(max(0.0, x_low - x_pad)*1.03, min(1.0, x_high + x_pad))
        ax.set_ylim(max(0.0, y_low - y_pad), y_high + y_pad)

    ax.annotate(
        'better',
        xy=(0.97, 0.07),
        xytext=(0.83, 0.20),
        xycoords='axes fraction',
        textcoords='axes fraction',
        fontsize=20,
        arrowprops={'arrowstyle': '->', 'linewidth': 1.8},
    )
    if SET_TITLE == True:
        ax.set_title(title, color=TEXT_GREEN, pad=20)
    ax.set_xlabel(xlabel,fontsize=24)
    ax.set_ylabel(ylabel, fontsize=24)
    _style_axis(ax)

def plot_degradation_matrix(fig, ax, uav_counts, ugv_counts, trained_combination, degradation_matrix, degradation_ci95_matrix,APPROACH_LABEL):
    ax.set_facecolor(PAPER_BG)
    image = ax.imshow(
        np.ma.masked_invalid(degradation_matrix),
        cmap=cmap,
        vmin=-50.0,
        vmax=0.0,
        aspect='equal',
    )

    for y in range(len(uav_counts)):
        for x in range(len(ugv_counts)):
            mean = degradation_matrix[y, x]
            ci95 = degradation_ci95_matrix[y, x]
            if np.isfinite(mean):
                annotation = f'{mean:+.1f} \n\u00b1 {ci95:.1f} \n(95% CI)' if np.isfinite(ci95) else f'{mean:+.1f} pp'
                color = 'white' if mean >= -15.0 else TEXT_DARK
            else:
                annotation = 'n/a'
                color = '#6f6758'
            ax.text(x, y, annotation, ha='center', va='center', color=color, fontsize=24, fontweight='bold')

    ax.set_xticks(np.arange(len(ugv_counts)), labels=ugv_counts)
    ax.set_yticks(np.arange(len(uav_counts)), labels=uav_counts)
    ax.set_xlabel('Number of UGVs', fontsize=20, color=TEXT_DARK, labelpad=10)
    ax.set_ylabel('Number of UAVs', fontsize=20, color=TEXT_DARK, labelpad=10)
    if SET_TITLE == True:
        ax.set_title(
            f'{APPROACH_LABEL} Confirmation AUC Change vs {trained_combination[0]} UAV / {trained_combination[1]} UGV',
            fontsize=22,
            fontweight='bold',
            color=TEXT_GREEN,
            pad=16,
        )
    ax.tick_params(axis='both', labelsize=20, colors=TEXT_DARK, length=0)
    ax.set_xticks(np.arange(-0.5, len(ugv_counts), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(uav_counts), 1), minor=True)
    ax.grid(which='minor', color=PAPER_BG, linewidth=3)
    ax.tick_params(which='minor', bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.05)
    colorbar.set_label('Confirm AUC change (percentage points)', fontsize=20, color=TEXT_DARK, labelpad=10)
    colorbar.ax.tick_params(labelsize=20, colors=TEXT_DARK)
    colorbar.outline.set_visible(False)
