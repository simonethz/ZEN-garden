"""
Cross-lambda evaluation figures for the profitability-bias-weight sweep.

Reads ``<dataset>/visualization/run_summary.csv`` and the per-run
``profitability_components.csv`` files in the timestamped subfolders.

Robustness notes (the original version assumed a single, clean 14-row sweep and
hard-coded a Linux sandbox output path):

* ``run_summary.csv`` accumulates one row per optimization across *repeated*
  sweeps (here: 38 rows = 5 sweeps). Rows are grouped into sweeps at every
  ``total_cost`` (= bias-disabled baseline) row, and the sweep covering the most
  distinct bias weights is selected for the cross-lambda figures. The bias
  weights (``LAMS``) are derived from that sweep, not hard-coded.
* The timestamped folders are created in the same chronological order as the
  CSV rows are appended, so folder[i] <-> row[i]. Folders are paired with the
  selected sweep by that index range instead of a fragile ``zip(folders, LAMS)``
  over mismatched lengths.
* Output paths are local and OS-independent.
"""
import re
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({'font.size': 10, 'axes.titlesize': 11, 'figure.dpi': 150})

# ---------- paths (override VIZ via env var ZEN_VIZ_DIR if needed) ----------
VIZ = Path(os.environ.get(
    "ZEN_VIZ_DIR",
    "D:/Students/ssambale_jwiegner/Crystal-Ball-small/visualization",
))
OUT = VIZ / "analysis"
OUT.mkdir(parents=True, exist_ok=True)

TECH_COLORS = {
    'battery': '#1f77b4', 'heat_pump': '#aec7e8', 'natural_gas_boiler': '#ff7f0e',
    'natural_gas_pipeline': '#ffbb78', 'natural_gas_turbine': '#2ca02c',
    'photovoltaics': '#98df8a', 'power_line': '#d62728', 'reservoir_hydro': '#ff9896',
    'run-of-river_hydro': '#9467bd', 'wind_offshore': '#c5b0d5', 'wind_onshore': '#8c564b',
}


def parse_lambda(label: str) -> float:
    """Bias weight from an ``optimization`` label; baseline (total_cost) -> 0.0."""
    m = re.search(r'bias_weight\s*=\s*([0-9.eE+-]+)', str(label))
    return float(m.group(1)) if m else 0.0


# ================= load & select the richest sweep =================
df_all = pd.read_csv(VIZ / 'run_summary.csv')
df_all['lam'] = df_all['optimization'].apply(parse_lambda)
df_all['is_baseline'] = df_all['optimization'].astype(str).str.strip() == 'total_cost'

# group rows into sweeps: a new sweep starts at every baseline row
df_all['sweep'] = df_all['is_baseline'].cumsum()

folders = sorted(glob.glob(str(VIZ / '2026-*')))
n = min(len(df_all), len(folders))
if len(df_all) != len(folders):
    print(f"  Warning: {len(df_all)} summary rows vs {len(folders)} folders; "
          f"pairing the first {n} by chronological order.")
df_all = df_all.iloc[:n].copy()
df_all['folder'] = folders[:n]

# pick the sweep with the most distinct bias weights (ties -> latest sweep)
sweep_sizes = df_all.groupby('sweep')['lam'].nunique()
best_sweep = sweep_sizes.sort_values(kind='stable').index[-1]
df = df_all[df_all['sweep'] == best_sweep].copy()
# within a sweep keep one row per lambda (last wins) and sort
df = df.drop_duplicates('lam', keep='last').sort_values('lam').reset_index(drop=True)

LAMS = df['lam'].tolist()
XTICK = [('0' if l == 0 else f'{l:g}') for l in LAMS]
X = np.arange(len(LAMS))
print(f"Selected sweep #{best_sweep} with {len(LAMS)} bias weights: {LAMS}")
print(f"Writing figures to {OUT}")

# ---- capacity-addition columns -> long form ----
cap_cols = [c for c in df.columns if c.startswith('cap|')]
df[cap_cols] = df[cap_cols].where(df[cap_cols].abs() > 1e-6, 0.0)
long = df.melt(id_vars='lam', value_vars=cap_cols, var_name='key', value_name='cap')
p = long['key'].str.split('|', expand=True)
long['tech'], long['node'], long['year'] = p[1], p[2], p[3].astype(int)

# ---- per-run profitability components, tagged with the run's lambda ----
allp = []
for _, r in df.iterrows():
    fp = Path(r['folder']) / 'profitability_components.csv'
    if not fp.exists():
        print(f"  Warning: missing {fp}")
        continue
    t = pd.read_csv(fp)
    t['lam'] = r['lam']
    allp.append(t)
allp = pd.concat(allp, ignore_index=True) if allp else pd.DataFrame()


# ================= FIG 1: cost + total additions =================
fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
ax = axes[0]
cost = df['total_cost'].values
rel = (cost / cost[0] - 1) * 100
ax.plot(X, rel, 'o-', color='#1f77b4', lw=1.8, ms=5)
# shade the regime split at the lambda where total additions stop tracking
# substitution and start growing (purely visual; only if we have >8 points)
if len(LAMS) > 8:
    ax.axvspan(-0.5, 7.5, color='#2ca02c', alpha=0.08)
    ax.axvspan(7.5, len(LAMS) - 0.5, color='#d62728', alpha=0.08)
    ymax = rel.max() if rel.max() > 0 else 1
    ax.text(3.5, ymax * 0.95, 'substitution regime', ha='center', color='#2ca02c', fontsize=9)
    ax.text((7.5 + len(LAMS)) / 2, ymax * 0.95, 'overbuild regime', ha='center',
            color='#d62728', fontsize=9)
ax.set_xticks(X); ax.set_xticklabels(XTICK, rotation=45)
ax.set_xlabel(r'bias weight $\lambda$')
ax.set_ylabel(r'total cost increase vs. $\lambda=0$ [%]')
ax.set_title('(a) Total system cost')
ax.grid(alpha=0.3)

ax = axes[1]
tot_tech = long.groupby(['tech', 'lam'])['cap'].sum().unstack('lam').reindex(columns=LAMS).fillna(0)
tot_tech = tot_tech.loc[tot_tech.sum(axis=1) > 0.5]
bottom = np.zeros(len(LAMS))
for tech, row in tot_tech.iterrows():
    ax.bar(X, row.values, bottom=bottom, color=TECH_COLORS.get(tech, 'gray'),
           label=tech, width=0.75, edgecolor='white', lw=0.3)
    bottom += row.values
ax.set_xticks(X); ax.set_xticklabels(XTICK, rotation=45)
ax.set_xlabel(r'bias weight $\lambda$'); ax.set_ylabel('total capacity additions [GW]')
ax.set_title('(b) Capacity additions (all nodes, all years)')
ax.legend(fontsize=7, ncol=2, loc='upper left')
fig.tight_layout()
fig.savefig(OUT / 'fig_lambda_sweep_overview.pdf')
fig.savefig(OUT / 'fig_lambda_sweep_overview.png', dpi=300)
plt.close(fig)

# ================= FIG 2: substitution heatmap =================
tot = long.groupby(['tech', 'node', 'lam'])['cap'].sum().unstack('lam').reindex(columns=LAMS).fillna(0)
delta = tot.sub(tot[LAMS[0]], axis=0)
delta = delta.loc[delta.abs().max(axis=1) > 0.5]
delta = delta.loc[delta.abs().max(axis=1).sort_values(ascending=False).index]
if len(delta):
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(delta) + 1.6))
    vmax = 50  # clip for readability; extremes annotated anyway
    im = ax.imshow(np.clip(delta.values, -vmax, vmax), cmap='RdBu_r',
                   vmin=-vmax, vmax=vmax, aspect='auto')
    for i in range(delta.shape[0]):
        for j in range(delta.shape[1]):
            v = delta.values[i, j]
            if abs(v) > 0.25:
                ax.text(j, i, f'{v:+.0f}' if abs(v) >= 3 else f'{v:+.1f}',
                        ha='center', va='center', fontsize=7,
                        color='white' if abs(v) > 0.6 * vmax else 'black')
    ax.set_xticks(range(len(LAMS))); ax.set_xticklabels(XTICK, rotation=45)
    ax.set_yticks(range(len(delta)))
    ax.set_yticklabels([f'{t} | {nd}' for t, nd in delta.index], fontsize=8)
    ax.set_xlabel(r'bias weight $\lambda$')
    ax.set_title(r'Change in total capacity additions vs. cost-optimal ($\lambda=0$) [GW]')
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label(r'$\Delta$ additions [GW] (clipped at $\pm$50)')
    fig.tight_layout()
    fig.savefig(OUT / 'fig_substitution_heatmap.pdf')
    fig.savefig(OUT / 'fig_substitution_heatmap.png', dpi=300)
    plt.close(fig)
else:
    print("  Skipping FIG 2: no (tech,node) pair changes additions by >0.5 GW.")

# ================= FIG 3: profitability contamination =================
fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
ax = axes[0]
for node, c in [('CH', '#1f77b4'), ('DE', '#ff7f0e'), ('IT', '#2ca02c')]:
    col_h = f'sp|heat|{node}|year_1'
    col_e = f'sp|electricity|{node}|year_1'
    if col_h in df:
        ax.plot(df['lam'], df[col_h].values * 1000, 'o-', color=c, label=f'heat {node}', ms=4)
    if col_e in df:
        ax.plot(df['lam'], df[col_e].values * 1000, 's--', color=c, alpha=0.5,
                label=f'electricity {node}', ms=3)
ax.set_xlabel(r'bias weight $\lambda$'); ax.set_ylabel('shadow price, year 1 [EUR/MWh]')
ax.set_title(r'(a) Year-1 shadow prices vs. $\lambda$')
ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)

ax = axes[1]
sel = [('heat_pump', 'IT', '#2ca02c'), ('natural_gas_boiler', 'IT', '#98df8a'),
       ('heat_pump', 'CH', '#1f77b4'), ('natural_gas_boiler', 'CH', '#aec7e8'),
       ('wind_onshore', 'DE', '#8c564b')]
if not allp.empty:
    for tech, node, c in sel:
        g = allp[(allp.decision_year == 1) & (allp.set_conversion_technologies == tech)
                 & (allp.set_nodes == node)].sort_values('lam')
        if len(g) < 2:
            continue
        sl = np.polyfit(g['lam'], g['profitability'], 1)[0]
        ax.plot(g['lam'], g['profitability'], 'o-', color=c, ms=4,
                label=f'{tech} {node} (slope {sl:.0f}/$\\lambda$)')
ax.set_xlabel(r'bias weight $\lambda$'); ax.set_ylabel('year-1 profitability [money/GW]')
ax.set_title(r'(b) Year-1 profitability vs. $\lambda$')
ax.legend(fontsize=7); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / 'fig_profitability_contamination.pdf')
fig.savefig(OUT / 'fig_profitability_contamination.png', dpi=300)
plt.close(fig)

# ================= FIG 4: overbuild threshold =================
if not allp.empty and {0, 1, 2}.issubset(set(allp['decision_year'].unique())):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharex=True)
    for ax, (tech, node) in zip(axes, [('heat_pump', 'IT'), ('natural_gas_boiler', 'IT')]):
        rewards, fcosts, adds = [], [], []
        for l in LAMS:
            g1 = allp[(allp.lam == l) & (allp.decision_year == 1)
                      & (allp.set_conversion_technologies == tech) & (allp.set_nodes == node)]
            g2 = allp[(allp.lam == l) & (allp.decision_year == 2)
                      & (allp.set_conversion_technologies == tech) & (allp.set_nodes == node)]
            rewards.append(l * g1['profitability'].iloc[0] if len(g1) else np.nan)
            fcosts.append((g2['capex'].iloc[0] + g2['fixed_opex'].iloc[0]) if len(g2) else np.nan)
            a = long[(long.lam == l) & (long.tech == tech) & (long.node == node) & (long.year == 2)]['cap']
            adds.append(a.iloc[0] if len(a) else 0.0)
        ax.plot(X, rewards, 'o-', color='#d62728', label=r'bias reward $\lambda\,\pi_{y1}$')
        ax.plot(X, fcosts, '--', color='black', label='annualized capex + fixed OPEX')
        ax.set_yscale('symlog', linthresh=1000)
        ax2 = ax.twinx()
        ax2.bar(X, adds, color='#aec7e8', alpha=0.55, width=0.7, label='year-2 additions')
        ax2.set_ylabel('year-2 additions [GW]', color='#5b87b5')
        ax.set_xticks(X); ax.set_xticklabels(XTICK, rotation=45)
        ax.set_xlabel(r'bias weight $\lambda$'); ax.set_ylabel('money / GW')
        ax.set_title(f'{tech} | {node}')
        ax.legend(fontsize=8, loc='upper left'); ax.grid(alpha=0.3)
    fig.suptitle(r'Overbuild onset: idle capacity becomes objective-profitable once '
                 r'$\lambda\,\pi > $ fixed cost', y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / 'fig_overbuild_threshold.pdf', bbox_inches='tight')
    fig.savefig(OUT / 'fig_overbuild_threshold.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
else:
    print("  Skipping FIG 4: need decision years {0,1,2} in profitability data.")

# ================= FIG 5 (new): cost vs. overbuild Pareto trade-off =================
# How much extra system cost does each lambda buy, and how much extra (idle)
# capacity does the bias force into the system? Total additions vs cost increase.
fig, ax = plt.subplots(figsize=(6.5, 5))
tot_add = long.groupby('lam')['cap'].sum().reindex(LAMS).values
overbuild = tot_add - tot_add[0]          # GW above the cost-optimal build-out
sc = ax.scatter(overbuild, rel, c=range(len(LAMS)), cmap='viridis', s=70, zorder=3)
ax.plot(overbuild, rel, '-', color='gray', alpha=0.5, zorder=2)
for xi, yi, lab in zip(overbuild, rel, XTICK):
    ax.annotate(lab, (xi, yi), textcoords='offset points', xytext=(5, 4), fontsize=8)
ax.set_xlabel(r'extra capacity built vs. $\lambda=0$ [GW]')
ax.set_ylabel(r'total cost increase vs. $\lambda=0$ [%]')
ax.set_title('Cost vs. overbuild trade-off of the profitability bias')
cb = fig.colorbar(sc, ax=ax); cb.set_label(r'$\lambda$ index (low $\to$ high)')
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / 'fig_cost_vs_overbuild.pdf')
fig.savefig(OUT / 'fig_cost_vs_overbuild.png', dpi=300)
plt.close(fig)

# ================= FIG 6 (new): how many investment decisions the bias flips =================
# Count (tech,node,year) cells whose capacity addition differs from the
# cost-optimal baseline by more than 0.5 GW, per lambda.
base = long[long.lam == LAMS[0]].set_index(['tech', 'node', 'year'])['cap']
flips, gw_moved = [], []
for l in LAMS:
    cur = long[long.lam == l].set_index(['tech', 'node', 'year'])['cap']
    d = (cur - base).reindex(base.index).fillna(0.0)
    flips.append(int((d.abs() > 0.5).sum()))
    gw_moved.append(float(d.abs().sum()))
fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.bar(X, flips, color='#4c72b0', width=0.7, label='# decisions changed (>0.5 GW)')
ax.set_xticks(X); ax.set_xticklabels(XTICK, rotation=45)
ax.set_xlabel(r'bias weight $\lambda$')
ax.set_ylabel('# (tech, node, year) cells changed', color='#4c72b0')
ax2 = ax.twinx()
ax2.plot(X, gw_moved, 'o-', color='#dd8452', label='total |GW| reallocated')
ax2.set_ylabel('total |GW| reallocated', color='#dd8452')
ax.set_title(r'Investment decisions altered by the bias vs. $\lambda$')
fig.tight_layout()
fig.savefig(OUT / 'fig_decisions_changed.pdf')
fig.savefig(OUT / 'fig_decisions_changed.png', dpi=300)
plt.close(fig)

# ================= FIG 7 (new): shadow-price contamination heatmap =================
# Year-1 shadow price of every (carrier, node) across lambda, relative to the
# cost-optimal price -> shows how broadly the bias inflates duals.
sp_cols = [c for c in df.columns if c.startswith('sp|') and c.endswith('|year_1')]
if sp_cols:
    sp = df.set_index('lam')[sp_cols].reindex(LAMS) * 1000  # EUR/MWh
    sp_delta = sp.sub(sp.iloc[0], axis=1).T
    sp_delta.index = [c.replace('sp|', '').replace('|year_1', '').replace('|', ' | ')
                      for c in sp_delta.index]
    sp_delta = sp_delta.loc[sp_delta.abs().max(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(9, 0.4 * len(sp_delta) + 1.6))
    vlim = np.nanpercentile(np.abs(sp_delta.values), 98) or 1.0
    im = ax.imshow(sp_delta.values, cmap='RdBu_r', vmin=-vlim, vmax=vlim, aspect='auto')
    ax.set_xticks(range(len(LAMS))); ax.set_xticklabels(XTICK, rotation=45)
    ax.set_yticks(range(len(sp_delta))); ax.set_yticklabels(sp_delta.index, fontsize=8)
    ax.set_xlabel(r'bias weight $\lambda$')
    ax.set_title(r'Year-1 shadow-price change vs. $\lambda=0$ [EUR/MWh]')
    cb = fig.colorbar(im, ax=ax, shrink=0.8); cb.set_label('$\\Delta$ shadow price [EUR/MWh]')
    fig.tight_layout()
    fig.savefig(OUT / 'fig_shadowprice_heatmap.pdf')
    fig.savefig(OUT / 'fig_shadowprice_heatmap.png', dpi=300)
    plt.close(fig)
else:
    print("  Skipping FIG 7: no year-1 shadow-price columns found.")

# ================= textual summary =================
print("\n================ ANALYSIS SUMMARY ================")
print(f"Bias weights analysed: {LAMS}")
print(f"Cost-optimal total cost (lambda=0): {cost[0]:,.0f}")
print(f"Max cost increase: {rel.max():.3f}% at lambda={LAMS[int(np.argmax(rel))]:g}")
print(f"Total additions at lambda=0: {tot_add[0]:.1f} GW; "
      f"max overbuild: {overbuild.max():.1f} GW at lambda={LAMS[int(np.argmax(overbuild))]:g}")
print(f"Most-altered lambda flips {max(flips)} (tech,node,year) decisions "
      f"({max(gw_moved):.1f} GW reallocated)")
print(f"\nFigures written to: {OUT}")
print("done")
