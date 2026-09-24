"""Graph-weighted analysis and plots for the checkpoint350 pilot."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

READOUT_CONDITIONS = ['C4', 'only_P1', 'only_P2', 'only_P3', 'only_P4']


def graph_values(rows, getter):
    values = defaultdict(list)
    for row in rows:
        value = getter(row)
        if value is not None and np.isfinite(value):
            values[row['graph_id']].append(float(value))
    return {g: float(np.mean(v)) for g, v in values.items()}


def estimate(values):
    x = np.array(list(values), dtype=float)
    if not len(x):
        return dict(mean=None, ci95=[None, None], n_graphs=0)
    rng = np.random.default_rng(20260917)
    means = x[rng.integers(0, len(x), (5000, len(x)))].mean(1)
    return dict(mean=float(x.mean()), ci95=np.quantile(means, [.025, .975]).tolist(), n_graphs=len(x))


def est(rows, getter):
    return estimate(graph_values(rows, getter).values())


def paired(rows, a, b, metric='forced_choice_correct'):
    return est(rows, lambda r: float(r['results'][a][metric]) - float(r['results'][b][metric]))


def auc(a, b):
    if not len(a) or not len(b):
        return None
    diff = np.asarray(a)[:, None] - np.asarray(b)[None, :]
    return float(np.mean((diff > 0) + .5 * (diff == 0)))


def representations(rows, graphs):
    records = []
    for row in rows:
        g = graphs[row['graph_id']]
        dist = {int(k): v for k, v in g['distances'].items()}
        nodes = sorted({v for e in g['edges'] for v in e})
        item = dict(graph_id=row['graph_id'], original_depth=row['original_depth'], steps={})
        for t in range(1, 5):
            frontier = [v for v in nodes if dist.get(v) == t]
            old = [v for v in nodes if dist.get(v, 99) < t]
            old_nonroot = [v for v in old if v != g['root']]
            future = [v for v in nodes if t < dist.get(v, 99) < 99]
            unreachable = [v for v in nodes if v not in dist]
            optimal = set(g['optimal'].get(str(t), []))
            offpath = [v for v in frontier if v not in optimal]
            sets = dict(frontier=frontier, old=old, old_nonroot=old_nonroot,
                        future=future, unreachable=unreachable, offpath_frontier=offpath)
            stats = {}
            for kind, source in [('logit', 'node_logits'), ('cosine', 'node_cosine')]:
                scores = np.array(row[source][t - 1])
                for name, vs in sets.items():
                    stats[f'{kind}_{name}'] = float(scores[vs].mean()) if vs else None
                for name in ('old', 'old_nonroot', 'future', 'unreachable'):
                    vs = sets[name]
                    stats[f'{kind}_frontier_vs_{name}_auc'] = auc(scores[frontier], scores[vs])
                    stats[f'{kind}_offpath_vs_{name}_auc'] = auc(scores[offpath], scores[vs])
                    stats[f'{kind}_frontier_minus_{name}'] = float(scores[frontier].mean() - scores[vs].mean()) if frontier and vs else None
                ranking = sorted(nodes, key=lambda v: scores[v], reverse=True)
                stats[f'{kind}_top1_in_frontier'] = float(ranking[0] in frontier) if frontier else None
                stats[f'{kind}_topk_frontier_recall'] = len(set(ranking[:len(frontier)]) & set(frontier)) / len(frontier) if frontier else None
                for d in range(5):
                    vs = [v for v in nodes if dist.get(v) == d]
                    stats[f'{kind}_depth{d}'] = float(scores[vs].mean()) if vs else None
            stats['frontier_count'] = len(frontier)
            stats['offpath_count'] = len(offpath)
            logits = np.array(row['node_logits'][t - 1])
            prob = np.exp(logits - logits.max())
            prob /= prob.sum()
            for name, vs in sets.items():
                stats[f'prob_mass_{name}'] = float(prob[vs].sum())
                stats[f'prob_per_node_{name}'] = float(prob[vs].mean()) if vs else None
            item['steps'][str(t)] = stats
        records.append(item)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    args = parser.parse_args()
    p = args.input
    metadata = json.loads((p / 'metadata.json').read_text())
    assert metadata['complete']
    rows = [json.loads(l) for l in (p / 'queries.jsonl').read_text(encoding='utf-8').splitlines()]
    originals = [json.loads(l) for l in (p / 'originals.jsonl').read_text(encoding='utf-8').splitlines()]
    graphs = json.loads((p / 'graphs.json').read_text())
    conditions = [c for c in READOUT_CONDITIONS if c in rows[0]['results']]
    if conditions != READOUT_CONDITIONS:
        raise ValueError(f'Missing expected readout conditions: {sorted(set(READOUT_CONDITIONS) - set(conditions))}')
    summary = dict(weighting='Mean within graph (nodes and two candidate orders), then equal graphs; 5000 graph bootstrap samples.',
                   native=est(originals, lambda r: r['results']['C4']['correct']),
                   native_n_correct=sum(r['results']['C4']['correct'] for r in originals),
                   native_n_queries=len(originals), by_depth={}, paired={}, subsets={}, representation={})
    for d in range(1, 5):
        bucket = [r for r in rows if r['target_depth'] == d]
        summary['by_depth'][str(d)] = dict(n_queries=len(bucket), conditions={
            c: {m: est(bucket, lambda r, c=c, m=m: r['results'][c][m]) for m in ('correct', 'forced_choice_correct', 'margin', 'outside_candidates')}
            for c in conditions})
        summary['paired'][str(d)] = {f'P{d}_minus_P{k}': paired(bucket, f'only_P{d}', f'only_P{k}') for k in range(1, 5) if k != d}
        for name, predicate in dict(first=lambda r: r['target_slot'] == 0,
                                    second=lambda r: r['target_slot'] == 1,
                                    leaf_matched=lambda r: r['leaf_status_matched'],
                                    modified_target=lambda r: r['target'] != r['original_target']).items():
            selected = [r for r in bucket if predicate(r)]
            summary['subsets'].setdefault(name, {})[str(d)] = {
                c: est(selected, lambda r, c=c: r['results'][c]['forced_choice_correct'])
                for c in conditions}
    records = representations(originals, graphs)
    (p / 'representation_per_query.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records), encoding='utf-8')
    common_ids = set(metadata['selected_graphs'])
    for cohort, selected in [('common_depth_graphs', [r for r in records if r['graph_id'] in common_ids]),
                              ('all_originals', records),
                              ('original_depth3', [r for r in records if r['original_depth'] == 3]),
                              ('original_depth4', [r for r in records if r['original_depth'] == 4])]:
        summary['representation'][cohort] = {str(t): {
            key: est(selected, lambda r, t=t, key=key: r['steps'][str(t)][key])
            for key in selected[0]['steps'][str(t)]} for t in range(1, 5)}
    (p / 'analysis.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Microsoft YaHei', 'DejaVu Sans'],
                         'font.size': 10, 'axes.unicode_minus': False, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout='constrained')
    matrices = [np.array([[summary['by_depth'][str(d)]['conditions'][c]['forced_choice_correct']['mean'] * 100 for c in conditions] for d in range(1, 5)]),
                np.array([[summary['representation']['common_depth_graphs'][str(t)][f'cosine_depth{d}']['mean'] for d in range(1, 5)] for t in range(1, 5)])]
    for i, (ax, mat) in enumerate(zip(axes, matrices)):
        im = (ax.imshow(mat, cmap='cividis', aspect='auto', vmin=0, vmax=100) if i == 0
              else ax.imshow(mat, cmap='RdBu_r', aspect='auto',
                             norm=TwoSlopeNorm(vmin=min(float(mat.min()), -.001), vcenter=0, vmax=float(mat.max()))))
        for y in range(mat.shape[0]):
            for x in range(mat.shape[1]):
                val = mat[y, x]
                red, green, blue, _ = im.cmap(im.norm(val))
                luminance = .2126 * red + .7152 * green + .0722 * blue
                ax.text(x, y, f'{val:.1f}' if i == 0 else f'{val:.3f}', ha='center', va='center',
                        color='white' if luminance < .48 else 'black')
        ax.set_yticks(range(4), ['1', '2', '3', '4'])
        ax.set_ylabel('目标最短距离（跳）' if i == 0 else '反馈向量 z_t 的时间步 t')
        ax.set_xticks(range(mat.shape[1]), ['C4'] + [str(c) for c in range(1, 5)] if i == 0 else [str(c) for c in range(1, 5)])
        ax.set_xlabel(['4-step baseline / 唯一保留的 latent KV 位置', '被测节点的 BFS 深度'][i])
        ax.set_title(['A  四步读出 × 问题深度（二选一 %）', 'B  原题同一轨迹的节点余弦分数'][i])
        fig.colorbar(im, ax=ax, shrink=.8)
    fig.suptitle(f'Checkpoint350：深度选择性验证\n{len(metadata["selected_graphs"])} 张图；{len(rows):,} 个深度查询；图等权平均', fontsize=14)
    fig.savefig(p / 'depth_frontier.png', dpi=180)
    fig.savefig(p / 'depth_frontier.pdf')
    plt.close(fig)
    print('native', summary['native'])
    for t, r in summary['representation']['common_depth_graphs'].items():
        print('representation', t, {k: r[k] for k in ['cosine_frontier_vs_old_nonroot_auc', 'cosine_frontier_vs_unreachable_auc', 'cosine_offpath_vs_old_nonroot_auc', 'cosine_topk_frontier_recall']})
    print('paired', json.dumps(summary['paired']))


if __name__ == '__main__':
    main()
