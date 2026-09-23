"""Graph-clustered summaries and figures for kv_frontier_experiment.py."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

METRICS = ['correct', 'pair_correct', 'target_prob', 'pair_prob', 'margin'] + [f'mass_{i}' for i in range(8)]
COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9', '#999999', '#444444']


class Aggregate:
    def __init__(self):
        self.sums = {}
        self.counts = defaultdict(int)

    def add(self, scope, condition, graph_id, value):
        key = scope, condition, graph_id
        arr = np.asarray(value, float)
        if key not in self.sums:
            self.sums[key] = np.zeros_like(arr)
        self.sums[key] += arr
        self.counts[key] += 1

    def means(self, scope, condition):
        return {g: total / self.counts[(s, c, g)] for (s, c, g), total in self.sums.items() if s == scope and c == condition}


def estimate(values):
    x = np.asarray(list(values), float)
    if not len(x):
        return dict(mean=None, ci95=[None, None], n_graphs=0)
    if x.ndim == 1:
        x = x[:, None]
    rng = np.random.default_rng(20260920)
    # Multinomial counts bootstrap whole graphs, keeping all queries together.
    weights = rng.multinomial(len(x), np.ones(len(x)) / len(x), size=2000) / len(x)
    means = weights @ x
    return dict(mean=np.nanmean(x, axis=0).tolist(), ci95=np.nanquantile(means, [.025, .975], axis=0).T.tolist(), n_graphs=len(x))


def scalar_est(values):
    value = estimate(values)
    return dict(mean=value['mean'][0], ci95=value['ci95'][0], n_graphs=value['n_graphs'])


def auc(a, b):
    if not len(a) or not len(b):
        return np.nan
    delta = a[:, None] - b[None, :]
    return float(((delta > 0) + .5 * (delta == 0)).mean())


def aggregate(folder, meta):
    agg, rep = Aggregate(), Aggregate()
    distances = {g['graph_id']: {int(k): v for k, v in g['distances'].items()} for g in json.loads((folder / 'graphs.json').read_text())}
    for kind in ['originals', 'queries']:
        with (folder / f'{kind}.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                g = row['graph_id']
                scope = 'original' if kind == 'originals' else f'depth{row["target_depth"]}'
                for cond, value in row['results'].items():
                    vector = [float(value[k]) for k in METRICS[:5]] + value['depth_mass']
                    agg.add(scope, cond, g, vector)
                    if kind == 'originals':
                        # Preserve each task's original 3/4-latent budget.
                        c = row['original_depth']
                        for mode in ['clean', 'markov1', 'feedback_only']:
                            for read in ['all', 'last']:
                                if cond == f'{mode}_C{c}_{read}':
                                    agg.add('native', f'{mode}_{read}', g, vector)
                    else:
                        if row['leaf_status_matched']:
                            agg.add(scope + '_leafmatched', cond, g, vector)
                        if row['target'] != row['original_target']:
                            agg.add(scope + '_modified', cond, g, vector)
                if kind == 'originals' and g in set(meta['selected_graphs']):
                    dist = distances[g]
                    for mode, arrays in row['representations'].items():
                        for t, (pp, ss) in enumerate(zip(arrays['probs'], arrays['cosine']), 1):
                            pp, ss = np.asarray(pp), np.asarray(ss)
                            for depth in range(1, 5):
                                positive = [v for v, d in dist.items() if d == depth]
                                negative = [v for v, d in dist.items() if d > 0 and d != depth]
                                values = [pp[positive].sum(), pp[positive].mean(), ss[positive].mean(), auc(ss[positive], ss[negative])]
                                rep.add(mode, f'z{t}_F{depth}', g, values)
    return agg, rep


def summarize(agg):
    out = {}
    groups = sorted({(s, c) for s, c, _ in agg.sums})
    for scope, cond in groups:
        vals = estimate(agg.means(scope, cond).values())
        out.setdefault(scope, {})[cond] = {m: dict(mean=vals['mean'][i], ci95=vals['ci95'][i], n_graphs=vals['n_graphs']) for i, m in enumerate(METRICS)}
    return out


def paired(agg, scope, a, b, metric):
    aa, bb = agg.means(scope, a), agg.means(scope, b)
    index = METRICS.index(metric)
    return scalar_est([(aa[g] - bb[g])[index] for g in sorted(aa.keys() & bb.keys())])


def save(fig, folder, name):
    for ext in ['png', 'svg']:
        fig.savefig(folder / f'{name}.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)


def heat(ax, matrix, title, xlabel, ylabel, xlabels, ylabels, vmin=0, vmax=100, cmap='cividis', fmt='.1f'):
    im = ax.imshow(matrix, vmin=vmin, vmax=vmax, aspect='auto', cmap=cmap)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel, xticks=range(len(xlabels)), yticks=range(len(ylabels)))
    ax.set_xticklabels(xlabels)
    ax.set_yticklabels(ylabels)
    for (i, j), v in np.ndenumerate(np.asarray(matrix)):
        rgb = im.cmap(im.norm(v))[:3]
        color = 'black' if np.dot(rgb, [.2126, .7152, .0722]) > .52 else 'white'
        ax.text(j, i, format(v, fmt), ha='center', va='center', color=color, fontsize=9)
    return im


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    args = parser.parse_args()
    folder = args.input
    meta = json.loads((folder / 'metadata.json').read_text())
    assert meta['complete']
    print('Aggregating', folder, flush=True)
    agg, reps = aggregate(folder, meta)
    stats = summarize(agg)
    repr_stats = {}
    for scope, cond in sorted({(s, c) for s, c, _ in reps.sums}):
        value = estimate(reps.means(scope, cond).values())
        repr_stats.setdefault(scope, {})[cond] = {m: dict(mean=value['mean'][i], ci95=value['ci95'][i], n_graphs=value['n_graphs']) for i, m in enumerate(['mass', 'per_node_mass', 'cosine', 'auc'])}
    steps, layers, ng = meta['steps'], meta['config']['n_layer'], len(meta['selected_graphs'])
    def stat(d, cond, metric='correct'):
        return stats[f'depth{d}'][cond][metric]
    def matrix(conds, metric='correct'):
        return np.array([[stat(d, c, metric)['mean'] * 100 for c in conds] for d in range(1, 5)])
    contrasts = {}
    comparisons = [
        ('drop2_C3_P3', 'drop2_once_C3_onlyP3', 'clean_C3_last'),
        ('drop1_C3_P3', 'drop1_once_C3_onlyP3', 'clean_C3_last'),
        ('drop2_Cfinal_P3', f'drop2_once_C{steps}_onlyP3', f'clean_C{steps}_onlyP3'),
        ('markov1_last', f'markov1_C{steps}_last', f'clean_C{steps}_last'),
        ('markov1_all', f'markov1_C{steps}_all', f'clean_C{steps}_all'),
    ]
    for d in range(1, 5):
        contrasts[str(d)] = {name: {m: paired(agg, f'depth{d}', a, b, m) for m in ['correct', 'pair_correct', 'target_prob']} for name, a, b in comparisons}
    # Within each graph, require an increase for both depth 2 AND depth 3.
    changes = {}
    for d in [2, 3]:
        a = agg.means(f'depth{d}', 'drop2_once_C3_onlyP3')
        b = agg.means(f'depth{d}', 'clean_C3_last')
        changes[d] = {g: (a[g] - b[g])[2] for g in a}
    joint = scalar_est([float(changes[2][g] > 0 and changes[3][g] > 0) for g in changes[2]])
    representation_changes = {}
    for d in range(1, 5):
        a = reps.means('drop2_once', f'z4_F{d}')
        b = reps.means('clean', f'z4_F{d}')
        representation_changes[str(d)] = {name: scalar_est([(a[g]-b[g])[i] for g in sorted(a)])
                                           for i, name in enumerate(['mass', 'per_node_mass', 'cosine', 'auc'])}
    graph_rows = json.loads((folder / 'graphs.json').read_text())
    graph_audit = {'n_graphs': len(graph_rows), 'expansion_revisits_old_frontier': {}, 'F3_expansion_hits_F2': 0}
    for t in range(1, 5):
        graph_audit['expansion_revisits_old_frontier'][str(t)] = 0
    for graph in graph_rows:
        dist = {int(k): v for k, v in graph['distances'].items()}
        for t in range(1, 5):
            revisited = {v for u, v in graph['edges'] if dist.get(u) == t and dist.get(v, 99) <= t}
            graph_audit['expansion_revisits_old_frontier'][str(t)] += bool(revisited)
            if t == 3:
                graph_audit['F3_expansion_hits_F2'] += any(dist[v] == 2 for v in revisited)
    summary = dict(metadata=meta, conditions=stats, representation=repr_stats, paired=contrasts,
                   simultaneous_F2_F3_target_probability_gain=joint,
                   h3_representation_change=representation_changes, graph_structure_audit=graph_audit,
                   native_paired={read: {m: paired(agg, 'native', f'markov1_{read}', f'clean_{read}', m)
                                        for m in ['correct', 'target_prob']} for read in ['all', 'last']},
                   uncertainty='2000 whole-graph bootstrap replicates, 95% percentile intervals; exploratory, no multiple-testing claim')
    (folder / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Microsoft YaHei', 'DejaVu Sans'],
                         'font.size': 10, 'axes.unicode_minus': False, 'svg.fonttype': 'none',
                         'axes.spines.top': False, 'axes.spines.right': False})
    title = f'{layers} 层 Transformer · {Path(meta["checkpoint"]).name} · {ng} 张图 / {meta["n_queries"]:,} 个查询'
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.7), layout='constrained')
    for ax, metric, label in zip(axs[:2], ['correct', 'pair_correct'], ['A  单个 KV 的完整词表准确率 (%)', 'B  单个 KV 的二选一准确率 (%)']):
        im = heat(ax, matrix([f'clean_C{steps}_onlyP{t}' for t in range(1, steps+1)], metric), label,
                  '唯一可读 latent KV 位置', '目标 BFS 深度', range(1, steps+1), range(1, 5))
        fig.colorbar(im, ax=ax, shrink=.8)
    arr = [[repr_stats['clean'][f'z{t}_F{d}']['auc']['mean'] for d in range(1, 5)] for t in range(1, steps+1)]
    im = heat(axs[2], arr, 'C  输入向量对各 frontier 的可分性', 'BFS 深度', '输入向量 z_t', range(1, 5), range(1, steps+1), 0, 1, fmt='.2f')
    fig.colorbar(im, ax=axs[2], shrink=.8, label='余弦 AUROC')
    fig.suptitle(title + '\n正常完整轨迹；A/B 保留题目 KV、答案绝对位置固定；C 比较其他可达深度节点')
    save(fig, folder, '01_baseline')

    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8), layout='constrained')
    configs = [('clean_C3_last', '正常 P3', COLORS[0]), ('drop2_once_C3_onlyP3', '构造 P3 时屏蔽 P2', COLORS[1]), ('drop1_once_C3_onlyP3', '构造 P3 时屏蔽 P1', COLORS[2])]
    for ax, metric, name in zip(axs[:2], ['correct', 'target_prob'], ['A  只读 P3 的完整词表准确率', 'B  只读 P3 的正确节点概率']):
        for ix, (cond, label, color) in enumerate(configs):
            vals = [stat(d, cond, metric) for d in range(1, 5)]
            means = np.array([v['mean'] for v in vals]) * 100
            ci = np.array([v['ci95'] for v in vals]) * 100
            ax.errorbar(np.arange(1, 5) + (ix-1)*.065, means, yerr=np.stack([means-ci[:,0], ci[:,1]-means]),
                        label=label, color=color, marker=['o','s','^'][ix], capsize=3)
        ax.set(title=name, xlabel='目标 BFS 深度', ylabel='%', xticks=range(1,5), ylim=(0,100))
        ax.legend(fontsize=8)
    names = ['F0', 'F1', 'F2', 'F3', 'F4', 'F>4', '不可达/未出现节点', '特殊 token']
    bottom = np.zeros(3)
    for k, name in enumerate(names):
        vals = [stats['original'][cond][f'mass_{k}']['mean'] * 100 for cond, _, _ in configs]
        axs[2].bar(range(3), vals, bottom=bottom, label=name, color=COLORS[k], edgecolor='white', linewidth=.3)
        bottom += vals
    axs[2].set(title='C  原题同一轨迹的答案概率质量', ylabel='完整词表概率 (%)', xticks=range(3), ylim=(0,100))
    axs[2].set_xticklabels(['正常 P3', '屏蔽 P2', '屏蔽 P1'])
    axs[2].legend(fontsize=7, loc='upper left', bbox_to_anchor=(1,1))
    fig.suptitle(title + '\n第三个位置之前的反馈完全相同；读答案时仅保留 P3 与题目 KV。误差线：图 bootstrap 95% CI')
    save(fig, folder, '02_P2_to_P3')

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.7), layout='constrained')
    for ax, mode, read, label in zip(axs, ['clean','markov1','markov1'], ['last','all','last'],
                                    ['A  正常推理，仅最后 KV 读出', 'B  前一个 KV 推理，全轨迹读出', 'C  前一个 KV 推理，仅最后 KV 读出']):
        im = heat(ax, matrix([f'{mode}_C{c}_{read}' for c in range(1, steps+1)]), label,
                  'latent 推理步数 C', '目标 BFS 深度', range(1,steps+1), range(1,5))
        fig.colorbar(im, ax=ax, shrink=.8, label='完整词表准确率 (%)')
    fig.suptitle(title + '\n题目 KV 始终保留；每个新位置只能读取前一个 latent KV，反馈向量持续传递')
    save(fig, folder, '03_markov')

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.7), layout='constrained')
    for d, color, marker in [(2,COLORS[0],'o'), (3,COLORS[1],'s')]:
        vals = [paired(agg, f'depth{d}', f'patch_P3_L{l}', 'clean_C3_last', 'target_prob') for l in range(1,layers+1)]
        y = np.array([v['mean'] for v in vals])*100
        ci = np.array([v['ci95'] for v in vals])*100
        axs[0].errorbar(range(1,layers+1), y, yerr=np.stack([y-ci[:,0],ci[:,1]-y]), color=color, marker=marker,
                        capsize=3, label=f'目标深度 {d}')
    axs[0].axhline(0, color='.5', lw=.8)
    axs[0].set(title='A  仅移植 P3 的一层 KV', xlabel='被替换的 Transformer 层', ylabel='正确节点概率变化 (百分点)', xticks=range(1,layers+1))
    axs[0].legend()
    variants = [('clean_all','正常推理\n全部 KV'), ('clean_last','正常推理\n最后 KV'), ('markov1_all','前一 KV 推理\n全部 KV 读出'), ('markov1_last','前一 KV 推理\n最后 KV 读出'), ('feedback_only_last','仅反馈推理\n最后 KV 读出')]
    vv = [stats['native'][k]['correct'] for k,_ in variants]
    y = np.array([v['mean'] for v in vv])*100
    ci = np.array([v['ci95'] for v in vv])*100
    axs[1].bar(range(len(vv)), y, color=COLORS[:len(vv)], yerr=np.stack([y-ci[:,0],ci[:,1]-y]), capsize=3)
    axs[1].set(title='B  原始测试题：原生 3/4 latent 预算', ylabel='完整词表准确率 (%)', ylim=(0,100), xticks=range(len(vv)))
    axs[1].set_xticklabels([label for _,label in variants], fontsize=8)
    for x, val in enumerate(y):
        axs[1].text(x,val+5,f'{val:.1f}',ha='center')
    fig.suptitle(title + '\nA：把屏蔽 P2 后产生的 P3 KV 移入正常缓存，只读 P3；B：图等权与 95% CI')
    save(fig, folder, '04_controls')

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.7), layout='constrained')
    for ax, mode, name in zip(axs, ['clean','drop2_once','markov1'], ['A  正常输入向量', 'B  构造 P3 时屏蔽 P2', 'C  持续只读前一 latent KV']):
        arr = [[repr_stats[mode][f'z{t}_F{d}']['mass']['mean']*100 for d in range(1,5)] for t in range(1,steps+1)]
        im = heat(ax, arr, name, 'BFS 深度', '输入向量 z_t', range(1,5), range(1,steps+1))
        fig.colorbar(im, ax=ax, shrink=.8, label='frontier 概率质量 (%)')
    fig.suptitle(title + '\n相同原题轨迹；直接使用共享 LM head 的完整词表 softmax。z3=h2 保持不变，z4=h3 可受干预影响')
    save(fig, folder, '05_latent_probability_mass')

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.7), layout='constrained')
    for ax, metric, label in zip(axs, ['correct', 'target_prob'], ['A  完整词表准确率 (%)', 'B  正确节点概率 (%)']):
        im = heat(ax, matrix([f'clean_C{c}_all' for c in range(steps+1)], metric), label,
                  '正常 latent 步数 C', '目标 BFS 深度', range(steps+1), range(1,5))
        fig.colorbar(im, ax=ax, shrink=.8)
    fig.suptitle(title + '\n全历史正常推理与答案读出；C=0 为零 latent，答案位置随 C 变化')
    save(fig, folder, '06_full_baseline_sweep')

    lines = [f'# {layers} 层模型的 frontier / KV 实验', '',
             f'权重：`{meta["checkpoint"]}`；SHA256：`{meta["checkpoint_sha256"]}`。', '',
             f'原题 {meta["n_original_queries"]//2} 张图 × 两种候选顺序；深度查询 {ng} 张图、{meta["n_queries"]} 题。模型冻结，FP32。', '',
             '## 先看 baseline', '',
             '| 原生预算条件 | 完整词表准确率 | 95% CI |', '|---|---:|---:|']
    for key,label in variants:
        v = stats['native'][key]['correct']
        lines.append(f'| {label.replace(chr(10)," / ")} | {v["mean"]*100:.2f}% | [{v["ci95"][0]*100:.2f}, {v["ci95"][1]*100:.2f}] |')
    lines += ['', f'固定 C={steps} 的全历史读出，各目标深度基线：', '', '| BFS 深度 | 全历史准确率 | 全 latent KV 遮蔽（答案位置不变） |', '|---|---:|---:|']
    for d in range(1,5):
        lines.append(f'| {d} | {stat(d,f"clean_C{steps}_all")["mean"]*100:.2f}% | {stat(d,f"clean_C{steps}_none")["mean"]*100:.2f}% |')
    lines += ['', '![full baseline](06_full_baseline_sweep.png)', '', '![baseline](01_baseline.png)', '',
              f'A/B 测量正常 {steps} 步轨迹中单个 KV 对不同深度答案的支持；C 使用原题相同轨迹，将输入向量与词表 embedding 做余弦比较，AUROC 的负例是其他可达深度。该指标未训练额外探针。', '',
              '## 屏蔽 P2，再构造 P3', '',
              '编号：P_t 是位置，z_t 是输入，h_t 是处理该位置后的顶层输出，z_(t+1)=h_t。干预保留已算好的 h2（即传给第三个位置的向量），在计算 P3 的全部 Transformer 层时屏蔽 P2。z3 因而完全不变；受影响的是 P3 的高层 KV、h3 及后续轨迹。第一层 P3 KV 在注意力前生成，因此必须不变，代码已逐元素验证。', '',
              '“只读 P3”默认保留题目 KV 和当前 [A] 的自注意力，其他 latent KV 全部屏蔽。结果另外保存屏蔽题目 KV 的只读 P3 对照。', '',
              '| 目标深度 | 正常 P3 准确率 | 屏蔽 P2 后 P3 准确率 | 正确节点概率变化及 95% CI |', '|---|---:|---:|---:|']
    for d in range(1,5):
        v = contrasts[str(d)]['drop2_C3_P3']['target_prob']
        lines.append(f'| {d} | {stat(d,"clean_C3_last")["mean"]*100:.2f}% | {stat(d,"drop2_once_C3_onlyP3")["mean"]*100:.2f}% | {v["mean"]*100:+.4f} [{v["ci95"][0]*100:+.4f}, {v["ci95"][1]*100:+.4f}] 个百分点 |')
    lines += ['', f'同一张图在 F2、F3 查询上平均正确节点概率同时增加的比例：{joint["mean"]*100:.2f}%（95% CI {joint["ci95"][0]*100:.2f}–{joint["ci95"][1]*100:.2f}%）。这是方向性描述，极小正变化也计入；不等于已证明两种信息存储在同一 KV。', '',
              '![P2 to P3](02_P2_to_P3.png)', '',
              '概率质量使用完整 40 token softmax，按最短距离分组，总和为 1。不可达部分包括词表内但图中未出现的节点。柱图使用原题同一轨迹；不同深度查询曲线会改变题面的候选节点，二者分开解释。', '',
              '## 连续前一 KV 推理', '',
              'markov1：计算 P_t 时保留题目、P_(t-1) 与当前位置，自注意力允许；更早 latent KV 被遮蔽。反馈从干预后的输出持续更新。all 读出允许访问所有历史缓存，last 读出只保留最后位置；all 是用来区分推理过程与答案读出需求的对照。feedback_only 连前一个 latent KV 也遮蔽，只靠反馈向量和题目继续。', '',
              '![markov](03_markov.png)', '', '![controls](04_controls.png)', '',
              '![latent probability mass](05_latent_probability_mass.png)', '',
              '输入向量的 softmax 质量与答案 token 的条件概率不是同一量。这里在同一原题轨迹上测量，避免把查询改写后的变化当成同一状态的内容。统计还包含每节点平均质量，防止 frontier 大小造成假象。', '',
              '## 与 proof 的关系', '',
              f'图结构核查：{graph_audit["n_graphs"]} 张图中，{graph_audit["F3_expansion_hits_F2"]} 张存在从 F3 扩展回已访问 F2 节点的边。', '',
              'proof 给出的是一种构造：F_(t+1)=N(F_t)\\R_t。若删掉历史里某层，下一输出最多可能保留本应被扣除的、恰好又被扩展到的旧节点；公式不直接预言完整 F2∪F3 被写进 P3。尤其在旧两层构造中，上层 KV 在该层注意力/MLP 更新前产生，反馈输出与同位置 KV 不可混称。六层训练模型也未被约束必须执行该构造。', '',
              '## 验证与解释边界', '',
              '正常 rollout 已与原生 Coconut.forward 比较 logits 与实际 latent 输入；所有被遮蔽位置注意力为零；全 latent 遮蔽等价于相同绝对位置的零 latent 对照；P2 干预前 h2 保持逐元素相同。详细误差见 metadata.json。', '',
              '统计先在每图内平均所有节点查询和两种候选顺序，再对图等权；95% CI 使用 2000 次图级 bootstrap。配对变化也以图为单位。单格为探索性结果，未做多重比较确认。', '',
              'frontier 选择性、原生答案读出与表示本身含有什么并非同一命题。只读 P3 仍可与题目相互作用；表示可解码和缓存移植效应不能直接证明严格 BFS 算法。两层旧模型与六层新模型训练续跑历史不同，跨模型差异仅是深度 pilot。', '',
              '所有完整概率、单层 KV 移植、持续屏蔽 P2、屏蔽 P1 对照均保存在逐题 JSONL；summary.json 包含按深度、原题、叶节点状态匹配、改写目标子集的图级统计。']
    (folder / 'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(dict(native_accuracy={k: v['correct']['mean'] for k,v in stats['native'].items()},
                          drop2_probability_delta={k: v['drop2_C3_P3']['target_prob'] for k,v in contrasts.items()},
                          joint_gain=joint), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
