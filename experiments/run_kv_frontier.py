"""Frozen symbol-Coconut frontier / KV-history experiments for arbitrary depth.

P_t is a latent input position; z_t is its input, h_t its top-layer output,
and z_(t+1)=h_t. Blocking P2 while computing P3 preserves h2 exactly.
All history masks preserve absolute positions, current-token self attention,
and (unless explicitly stated) the entire question prefix.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from run_depth_selectivity import DATA_DIR, ROOT, load, make_queries, sha


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def history_slots(mode, step):
    """1-based visible past slots when computing P_step."""
    slots = list(range(1, step))
    if mode == 'drop2_once' and step == 3:
        slots.remove(2)
    elif mode == 'drop1_once' and step == 3:
        slots.remove(1)
    elif mode == 'drop2_persistent' and step >= 3:
        slots.remove(2)
    elif mode == 'markov1':
        slots = slots[-1:]
    elif mode == 'feedback_only':
        slots = []
    return slots


class Runner:
    def __init__(self, model):
        self.model = model
        self.base = model.base_causallm
        self.device = next(model.parameters()).device
        self.audits = {}

    def call(self, kv, p, count, emb=None, keep=None, position=None,
             prompt=True, audit=False):
        b = kv[0][0].shape[0]
        keep = list(range(1, count + 1)) if keep is None else keep
        mask = torch.ones(b, p + count + 1, device=self.device, dtype=torch.long)
        mask[:, p:p + count] = 0
        for slot in keep:
            assert 1 <= slot <= count
            mask[:, p + slot - 1] = 1
        if not prompt:
            mask[:, :p] = 0
        kwargs = dict(inputs_embeds=emb) if emb is not None else dict(
            input_ids=torch.full((b, 1), 37, device=self.device))
        out = self.base(**kwargs, past_key_values=kv, attention_mask=mask,
                        position_ids=torch.full((b, 1), p + (count if position is None else position), device=self.device),
                        output_hidden_states=emb is not None, output_attentions=audit,
                        use_cache=emb is not None)
        if audit and bool((mask == 0).any()):
            maximum = max(float(a.masked_select((mask == 0)[:, None, None, :]).abs().max()) for a in out.attentions)
            assert maximum < 1e-7
            self.audits['max_masked_attention'] = max(self.audits.get('max_masked_attention', 0), maximum)
        return out

    def rollout(self, prefix, p, mode, steps, audit):
        states = [prefix]
        for t in range(1, steps + 1):
            prev = states[-1]
            state = self.call(prev.past_key_values, p, t - 1,
                              emb=prev.hidden_states[-1][:, -1:],
                              keep=history_slots(mode, t), audit=audit)
            states.append(state)
        return states

    def read(self, state, p, count, keep=None, position=None, prompt=True, audit=False):
        return self.call(state.past_key_values, p, count, keep=keep,
                         position=position, prompt=prompt, audit=audit).logits[:, -1]


def values(logits, rows, distances):
    probs = logits.softmax(-1).cpu().numpy()
    scores = logits.cpu().numpy()
    result = []
    for pp, ll, row in zip(probs, scores, rows):
        dist = distances[row['graph_id']]
        masses = [float(pp[[v for v, d in dist.items() if d == depth]].sum()) for depth in range(5)]
        other_reachable = [v for v, d in dist.items() if d > 4]
        outside = [v for v in range(31) if v not in dist]
        masses += [float(pp[other_reachable].sum()), float(pp[outside].sum()), float(pp[31:].sum())]
        assert abs(sum(masses) - 1) < 1e-5
        target, neg = row['target'], row['neg_target']
        result.append(dict(correct=int(ll.argmax()) == target,
                           pair_correct=bool(ll[target] > ll[neg]),
                           target_prob=float(pp[target]), margin=float(ll[target] - ll[neg]),
                           pair_prob=float(torch.sigmoid(torch.tensor(float(ll[target] - ll[neg])))),
                           depth_mass=masses, probabilities=pp.tolist()))
    return result


@torch.inference_mode()
def evaluate(runner, tok, rows, distances, output, steps, batch_size, representation=False):
    groups = defaultdict(list)
    for row in rows:
        groups[row['prompt_length']].append(row)
    done, start = 0, time.time()
    with output.open('w', encoding='utf-8') as stream:
        for p, group in sorted(groups.items()):
            for offset in range(0, len(group), batch_size):
                rr = group[offset:offset + batch_size]
                b = len(rr)
                ids = torch.tensor([tok.encode(r['prompt'], add_special_tokens=False) for r in rr], device=runner.device)
                assert ids.shape == (b, p)
                prefix = runner.base(input_ids=ids, attention_mask=torch.ones_like(ids),
                                     position_ids=torch.arange(p, device=runner.device)[None].expand(b, -1),
                                     output_hidden_states=True, use_cache=True)
                audit = done == 0
                clean = runner.rollout(prefix, p, 'clean', steps, audit)
                results = {}
                def record(name, logits):
                    results[name] = values(logits, rr, distances)
                for c in range(steps + 1):
                    record(f'clean_C{c}_all', runner.read(clean[c], p, c))
                    if c:
                        record(f'clean_C{c}_last', runner.read(clean[c], p, c, [c], audit=audit))
                for t in range(1, steps + 1):
                    record(f'clean_C{steps}_onlyP{t}', runner.read(clean[-1], p, steps, [t], audit=audit))
                    record(f'clean_C{steps}_dropP{t}', runner.read(clean[-1], p, steps, [s for s in range(1, steps + 1) if s != t]))
                none = runner.read(clean[-1], p, steps, [], audit=audit)
                zero = runner.read(prefix, p, 0, position=steps)
                torch.testing.assert_close(none, zero, atol=5e-5, rtol=5e-5)
                record(f'clean_C{steps}_none', none)
                record('clean_C3_onlyP3_no_prompt', runner.read(clean[3], p, 3, [3], prompt=False, audit=audit))
                if audit:
                    for c in sorted(set([3, 4, steps])):
                        official_ids = torch.cat([ids[:1], torch.full((1, c), 33, device=runner.device), torch.full((1, 1), 37, device=runner.device)], 1)
                        official = runner.model(official_ids, torch.ones_like(official_ids), official_ids,
                                                torch.arange(p + c + 1, device=runner.device)[None], inference_only=True)
                        own = runner.read(clean[c], p, c)
                        diff = float((official.logits[0, -1] - own[0]).abs().max())
                        torch.testing.assert_close(official.logits[0, -1], own[0], atol=5e-5, rtol=5e-5)
                        inputs = torch.cat([s.hidden_states[-1][:1, -1:] for s in clean[:c]], 1)
                        torch.testing.assert_close(official.inputs_embeds[:1, p:p+c], inputs, atol=5e-5, rtol=5e-5)
                        runner.audits[f'official_C{c}_max_logit_diff'] = diff
                reprs = {}
                def representation_record(name, states):
                    if not representation:
                        return
                    zs = torch.cat([s.hidden_states[-1][:, -1:] for s in states[:-1]], dim=1)
                    head = runner.model.embedding.weight
                    reprs[name] = dict(probs=(zs @ head.T).softmax(-1).cpu().numpy(),
                                       cosine=(F.normalize(zs, dim=-1) @ F.normalize(head, dim=-1).T).cpu().numpy())
                representation_record('clean', clean)
                for mode in ['drop2_once', 'drop1_once', 'drop2_persistent', 'markov1', 'feedback_only']:
                    states = runner.rollout(prefix, p, mode, steps, audit)
                    if mode.startswith('drop'):
                        # The input to P3 and all earlier caches are identical.
                        torch.testing.assert_close(states[2].hidden_states[-1], clean[2].hidden_states[-1], rtol=0, atol=0)
                        if audit:
                            runner.audits['intervention_preserves_h2_exactly'] = True
                        for c in sorted(set([3, steps])):
                            visible = [s for s in range(1, c + 1) if not (mode == 'drop2_persistent' and s == 2)]
                            record(f'{mode}_C{c}_all', runner.read(states[c], p, c, visible))
                            record(f'{mode}_C{c}_onlyP3', runner.read(states[c], p, c, [3], audit=audit))
                        if mode == 'drop2_once':
                            record('drop2_once_C3_onlyP3_no_prompt', runner.read(states[3], p, 3, [3], prompt=False, audit=audit))
                            # Transfer one changed P3 layer into the otherwise clean cache.
                            for layer in range(runner.base.config.n_layer):
                                patched = list(clean[3].past_key_values)
                                source = states[3].past_key_values[layer]
                                target = patched[layer]
                                patched[layer] = tuple(torch.cat([a[:, :, :p+2], z[:, :, p+2:p+3]], dim=-2) for a, z in zip(target, source))
                                logits = runner.call(tuple(patched), p, 3, keep=[3]).logits[:, -1]
                                record(f'patch_P3_L{layer+1}', logits)
                            if audit:
                                # First layer K/V is computed before history attention.
                                for a, z in zip(clean[3].past_key_values[0], states[3].past_key_values[0]):
                                    torch.testing.assert_close(a[:, :, -1], z[:, :, -1], atol=0, rtol=0)
                                runner.audits['P3_layer1_KV_unchanged_exactly'] = True
                    else:
                        for c in range(1, steps + 1):
                            record(f'{mode}_C{c}_all', runner.read(states[c], p, c))
                            record(f'{mode}_C{c}_last', runner.read(states[c], p, c, [c], audit=audit))
                    representation_record(mode, states)
                    del states
                for j, row in enumerate(rr):
                    item = dict(row, results={k: v[j] for k, v in results.items()})
                    if representation:
                        item['representations'] = {k: {a: v[j].tolist() for a, v in value.items()} for k, value in reprs.items()}
                    stream.write(json.dumps(item, ensure_ascii=False) + '\n')
                done += b
                if done % 128 < b or done == len(rows):
                    stream.flush()
                    print(f'{output.stem}: {done}/{len(rows)} ({time.time()-start:.1f}s)', flush=True)
    return dict(runner.audits)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--graphs', type=int, default=0, help='0: every graph with depths 1..4')
    parser.add_argument('--original-graphs', type=int, default=0)
    parser.add_argument('--steps', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=24)
    parser.add_argument('--seed', type=int, default=20260917)
    args = parser.parse_args()
    assert args.steps >= 4
    if args.output.exists():
        raise FileExistsError('Use a new output directory')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dataset = DATA_DIR / 'prosqa_test_graph_4_coconut_shuffled_with_bfs.json'
    data = json.loads(dataset.read_text(encoding='utf-8'))
    queries, originals, distances, selected, eligible = make_queries(data, args.graphs, args.seed)
    if args.original_graphs:
        originals = [q for q in originals if q['graph_id'] < args.original_graphs]
    model, tok = load(args.checkpoint, args.config)
    args.output.mkdir(parents=True)
    meta = dict(complete=False, started=datetime.now(timezone.utc).isoformat(),
                checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=sha(args.checkpoint),
                dataset_sha256=sha(dataset), script_sha256=sha(Path(__file__)),
                loader_sha256=sha(Path(__file__).with_name('run_depth_selectivity.py')),
                config=model.base_causallm.config.to_dict(),
                seed=args.seed, selected_graphs=selected, eligible_graphs=len(eligible),
                n_queries=len(queries), n_original_queries=len(originals), steps=args.steps,
                torch=torch.__version__, device=torch.cuda.get_device_name(),
                definition='z_t=input at P_t; h_t=output at P_t; z_(t+1)=h_t. drop2 at P3 preserves h2.',
                question_KV='Always visible except explicitly named no_prompt readouts',
                depth_mass_labels=['F0','F1','F2','F3','F4','F>4','unreachable_node_ids','special_tokens'],
                statistics='Graph clustered, candidate orders paired; interventions never retrain model')
    dump(args.output / 'metadata.json', meta)
    dump(args.output / 'graphs.json', [dict(graph_id=i, distances=d, edges=s['edges'], root=s['root']) for i, (d, s) in enumerate(zip(distances, data))])
    runner = Runner(model)
    print(f'layers={model.base_causallm.config.n_layer}; originals={len(originals)}; queries={len(queries)}; graphs={len(selected)}', flush=True)
    meta['original_audits'] = evaluate(runner, tok, originals, distances, args.output / 'originals.jsonl', args.steps, args.batch_size, True)
    meta['query_audits'] = evaluate(runner, tok, queries, distances, args.output / 'queries.jsonl', args.steps, args.batch_size)
    meta.update(complete=True, finished=datetime.now(timezone.utc).isoformat())
    dump(args.output / 'metadata.json', meta)


if __name__ == '__main__':
    main()
