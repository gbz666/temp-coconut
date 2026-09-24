"""Checkpoint350 pilot: depth-selective latent KV readout and node scores.

No training. Original and counterfactual questions use a four-step trajectory.
Counterfactual questions retain graph/root/negative and enumerate depths 1..4.
z_t is the vector FED INTO latent slot P_t, t=1..4. In this curriculum the
root-position output predicts neighbor_1, so z_1 is compared against F_1.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

os.environ['USE_TF'] = '0'
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoConfig, AutoModelForCausalLM
from prepare_eval_queries import DEFAULT_OUTPUT as DEFAULT_PREPARED

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC_DIR = ROOT / 'src'
CONFIG_DIR = ROOT / 'configs'
DATA_DIR = ROOT / 'data'
CHECKPOINT_DIR = ROOT / 'checkpoints'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def choose_device(explicit=None):
    if explicit:
        return explicit
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def device_name(device):
    if str(device).startswith('cuda'):
        return torch.cuda.get_device_name()
    if str(device) == 'mps':
        return 'mps'
    return 'cpu'


def load(checkpoint, config_path=None, device='cuda'):
    tokenizer = module('emergence_tokenizer', SRC_DIR / 'stokenizer.py').STokenizer()
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    # Layer count is encoded in the state dict; head count is not. Without an
    # explicit config retain the established 8-head symbol-model family.
    import re
    layers = sorted({int(m.group(1)) for k in state
                     if (m := re.match(r'base_causallm\.transformer\.h\.(\d+)\.', k))})
    if not layers or layers != list(range(len(layers))):
        raise ValueError('Expected a Coconut GPT-2 state dict with contiguous layers')
    config = AutoConfig.from_pretrained(str(config_path or CONFIG_DIR / 'model_2layer.json'))
    if config_path is not None and config.n_layer != len(layers):
        raise ValueError('Config layer count does not match checkpoint')
    config.n_layer = len(layers)
    base = AutoModelForCausalLM.from_config(config, attn_implementation='eager')
    model = module('emergence_coconut', SRC_DIR / 'coconut.py').Coconut(base, 33, 31, 32, 38)
    model.load_state_dict(state, strict=True)
    model.eval()
    model.to(device)
    assert base.lm_head.weight.data_ptr() == model.embedding.weight.data_ptr()
    return model, tokenizer


def read_jsonl(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def metrics(logits, rows):
    arr = logits.float().cpu().numpy()
    result = []
    for values, row in zip(arr, rows):
        token = int(values.argmax())
        margin = float(values[row['target']] - values[row['neg_target']])
        result.append(dict(token=token, correct=token == row['target'],
                           forced_choice_correct=margin > 0, margin=margin,
                           outside_candidates=token not in (row['target'], row['neg_target'])))
    return result


def decode(base, kv, p, count, batch, keep=None, audit=False, device='cuda'):
    mask = torch.ones(batch, p + count + 1, device=device, dtype=torch.long)
    if keep is not None:
        mask[:, p:p + count] = 0
        for slot in keep:
            mask[:, p + slot] = 1
    out = base(input_ids=torch.full((batch, 1), 37, device=device), past_key_values=kv,
               attention_mask=mask, position_ids=torch.full((batch, 1), p + count, device=device),
               output_attentions=audit, use_cache=False)
    if audit and bool((mask == 0).any()):
        for a in out.attentions:
            assert float(a.masked_select((mask == 0)[:, None, None, :]).abs().max()) < 1e-7
    return out.logits[:, -1]


@torch.inference_mode()
def evaluate(model, tokenizer, rows, output, representation=False, batch_size=24, device='cuda'):
    base = model.base_causallm
    groups = defaultdict(list)
    for row in rows:
        groups[row['prompt_length']].append(row)
    done = 0
    audits = []
    start = time.time()
    with output.open('w', encoding='utf-8') as stream:
        for p, group in sorted(groups.items()):
            for offset in range(0, len(group), batch_size):
                batch_rows = group[offset:offset + batch_size]
                b = len(batch_rows)
                ids = torch.tensor([tokenizer.encode(r['prompt'], add_special_tokens=False) for r in batch_rows], device=device)
                assert ids.shape == (b, p)
                prefix = base(input_ids=ids, attention_mask=torch.ones_like(ids),
                              position_ids=torch.arange(p, device=device)[None].expand(b, -1),
                              output_hidden_states=True, use_cache=True)
                out, kv = prefix, prefix.past_key_values
                results, vectors = {}, []
                for count in range(4):
                    z = out.hidden_states[-1][:, -1:]
                    if representation:
                        vectors.append(z[:, 0])
                    out = base(inputs_embeds=z, past_key_values=kv,
                               attention_mask=torch.ones(b, p + count + 1, device=device, dtype=torch.long),
                               position_ids=torch.full((b, 1), p + count, device=device),
                               output_hidden_states=True, use_cache=True)
                    kv = out.past_key_values
                logits = decode(base, kv, p, 4, b, device=device)
                results['C4'] = metrics(logits, batch_rows)
                for slot in range(4):
                    slot_logits = decode(base, kv, p, 4, b, keep=[slot], audit=done == 0, device=device)
                    results[f'only_P{slot + 1}'] = metrics(slot_logits, batch_rows)
                if done == 0:
                    official_ids = torch.cat([ids[:1], torch.tensor([[33] * 4 + [37]], device=device)], dim=1)
                    official = model(official_ids, torch.ones_like(official_ids), official_ids,
                                     torch.arange(official_ids.shape[1], device=device)[None], inference_only=True)
                    torch.testing.assert_close(official.logits[0, -1], logits[0], atol=3e-5, rtol=3e-5)
                    if representation:
                        torch.testing.assert_close(official.inputs_embeds[0, p:p + 4], torch.stack(vectors, dim=1)[0], atol=3e-5, rtol=3e-5)
                    audits.append(dict(latent_count=4, official_max_logit_diff=float((official.logits[0, -1] - logits[0]).abs().max())))
                if representation:
                    zs = torch.stack(vectors, dim=1)
                    emb = model.embedding.weight.detach()
                    scores = (zs @ emb.T).cpu().tolist()
                    cosine = (F.normalize(zs, dim=-1) @ F.normalize(emb, dim=-1).T).cpu().tolist()
                for j, row in enumerate(batch_rows):
                    record = {**row, 'results': {k: v[j] for k, v in results.items()}}
                    if representation:
                        record['node_logits'] = scores[j]
                        record['node_cosine'] = cosine[j]
                    stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                done += b
                if done % 100 < b or done == len(rows):
                    stream.flush()
                    print(f'{output.stem}: {done}/{len(rows)}; {time.time() - start:.1f}s', flush=True)
    return audits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, default=DEFAULT_PREPARED)
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT_DIR / '2layer/checkpoint_350')
    parser.add_argument('--config', type=Path, help='Explicit model JSON; otherwise infer depth in the standard 8-head family')
    parser.add_argument('--batch-size', type=int, default=24)
    parser.add_argument('--device', choices=['cuda', 'mps', 'cpu'], default=None)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Choose a new output directory; existing results are never overwritten.')
    torch.set_num_threads(4)
    selection = json.loads((args.prepared / 'selection.json').read_text(encoding='utf-8'))
    torch.manual_seed(selection['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dataset = DATA_DIR / 'prosqa_test_graph_4_coconut_shuffled_with_bfs.json'
    if selection['dataset_sha256'] != sha(dataset):
        raise ValueError('Prepared queries do not match the current test dataset')
    queries = read_jsonl(args.prepared / 'queries.jsonl')
    originals = read_jsonl(args.prepared / 'originals.jsonl')
    selected = selection['selected_graphs']
    if len(queries) != selection['n_queries'] or len(originals) != selection['n_original_queries']:
        raise ValueError('Prepared query counts do not match selection.json')
    device = choose_device(args.device)
    model, tokenizer = load(args.checkpoint, args.config, device=device)
    args.output.mkdir(parents=True)
    metadata = dict(complete=False, started_utc=datetime.now(timezone.utc).isoformat(),
                    checkpoint=str(args.checkpoint), checkpoint_sha256=sha(args.checkpoint),
                    dataset=str(dataset), dataset_sha256=sha(dataset), script_sha256=sha(Path(__file__)),
                    seed=selection['seed'],
                    torch=torch.__version__, transformers=transformers.__version__,
                    device=device_name(device), dtype='float32', model_layers=model.base_causallm.config.n_layer,
                    selected_graphs=selected, eligible_graph_count=selection['eligible_graph_count'], n_queries=len(queries),
                    n_original_queries=len(originals), bootstrap_unit='graph',
                    training=False, data_node_labels='pre-shuffled official test labels',
                    latent_definition='z_t is feedback input to P_t; z_1 is root-position output, compared with F_1',
                    readout='Prompt always visible. only_P uses four-step trajectory and fixed answer position P+4.',
                    sweep='C4 full readout plus only_P1..only_P4 single-latent-KV readout.',
                    representation='Original query-conditioned trajectory only; no per-node prompt changes; tied LM head dot products and cosine.')
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    (args.output / 'graphs.json').write_bytes((args.prepared / 'graphs.json').read_bytes())
    print(f'Originals {len(originals)}; depth queries {len(queries)} on {len(selected)} graphs', flush=True)
    metadata['original_audits'] = evaluate(model, tokenizer, originals, args.output / 'originals.jsonl', True, args.batch_size, device=device)
    metadata['query_audits'] = evaluate(model, tokenizer, queries, args.output / 'queries.jsonl', False, args.batch_size, device=device)
    metadata.update(complete=True, finished_utc=datetime.now(timezone.utc).isoformat(), single_slot_mask_audit=True)
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print('Complete: ' + str(args.output), flush=True)


if __name__ == '__main__':
    main()
