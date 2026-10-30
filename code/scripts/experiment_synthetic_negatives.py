"""Bounded weak-label augmentation; train/val only, no holdout access."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.data.acceptance import sha256, validate_lock
from src.models.interaction_features import interaction_features
from src.models.regression_head import save_head, load_head
from src.training.synthetic_negatives import negative_indices
from src.training.train_regression import fit_heads
from src.evaluation.metrics import pairwise_accuracy


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    manifest = a.root/'data/manifest.jsonl'
    validate_lock(a.root/'data/lock.json', manifest)
    extraction = json.loads((a.root/'extraction/result.json').read_text(encoding='utf-8'))
    embedding = a.root/'extraction/embeddings.npz'
    assert sha256(embedding) == extraction['embeddings_sha256']
    assert sha256(manifest) == extraction['manifest_sha256']
    records = [json.loads(s) for s in manifest.read_text(encoding='utf-8').splitlines()]
    partitions = {}
    with np.load(embedding, allow_pickle=False) as data:
        assert data['pair_ids'].tolist() == [r['pair_id'] for r in records]
        for split in ('train', 'val'):
            ix = [i for i, r in enumerate(records) if r['split'] == split]
            partitions[split] = (data['text'][ix], data['image'][ix],
                np.array([records[i]['target'] for i in ix]),
                [records[i]['prompt_group'] for i in ix])
    protocol = {'scope': 'weak_label_development_not_release', 'seeds': [42,43,44],
        'fractions': [0.1,0.25,0.5], 'feature_map': 'hybrid', 'negative_target': 1,
        'negative_label_source': 'synthetic_assumption_not_human',
        'negative_seed': 1701, 'maximum_text_similarity': 0.5,
        'holdout': 'NOT EVALUATED', 'manifest_sha256': sha256(manifest),
        'embeddings_sha256': sha256(embedding),
        'provisional_gate': {'validation_mae_max_each_seed': 0.80377728737,
            'external20_seen_mae_max_each_seed': 1.165216147104,
            'all_three_known_ordering_checks': True},
        'deployment': 'No automatic promotion; independent review remains required'}
    (a.output/'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    t, im, y, groups = partitions['train']
    source, replacement = negative_indices(t, groups)
    with (a.output/'synthetic_pairs.json').open('x', encoding='utf-8') as f:
        json.dump({'source_train_indices':source.tolist(),
            'replacement_train_indices':replacement.tolist(),
            'label_source':'synthetic_assumption_not_human'}, f)
    x = interaction_features(t, im, 'hybrid')
    vt, vi, vy, vg = partitions['val']
    vx = interaction_features(vt, vi, 'hybrid')
    runs = {}
    for fraction in protocol['fractions']:
        n = min(int(len(y)*fraction), len(source))
        nx = interaction_features(t[replacement[:n]], im[source[:n]], 'hybrid')
        for seed in protocol['seeds']:
            models, report = fit_heads(np.concatenate([x,nx]),
                np.concatenate([y,np.ones(n)]), vx, vy, seed=seed)
            folder = a.output/f'negative{fraction}-seed{seed}'
            folder.mkdir()
            for kind, model in models.items():
                path = folder/f'{kind}.pt'
                save_head(model,path)
                before = model.predict(vx)
                delta = float(np.max(np.abs(load_head(path).predict(vx)-before)))
                assert delta < 1e-5
                report.setdefault('checkpoints',{})[kind] = {'sha256':sha256(path),'reload_delta':delta}
                report['metrics'][kind]['pairwise'] = pairwise_accuracy(vy,before,vg)
            report.update(feature_map='hybrid', seed=seed, synthetic_count=n,
                          synthetic_fraction=fraction, label_source=protocol['negative_label_source'])
            (folder/'result.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
            runs[folder.name] = {'metrics':report['metrics'],'best_epoch':report['best_epoch'],
                                 'synthetic_count':n}
            print(folder.name, report['metrics']['mlp'],flush=True)
    (a.output/'summary.json').write_text(json.dumps({'protocol':protocol,'runs':runs},indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
