"""Evaluate a frozen candidate once; never fit or select using test results."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.data.acceptance import sha256, validate_lock
from src.evaluation.metrics import regression_metrics, pairwise_accuracy, spearman
from src.models.regression_head import features, load_head


def verify_files(root, protocol):
    root = Path(root).resolve()
    for name, digest in protocol['files_sha256'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or sha256(path) != digest:
            raise ValueError(f'Frozen input changed: {name}')
    if protocol['selected'] != 'mlp' or protocol['seed'] != 42 or protocol['min_label_gap'] != 0.5:
        raise ValueError('Unexpected evaluation policy')


def evaluate(root, protocol_path, output):
    root, protocol_path, output = Path(root), Path(protocol_path), Path(output)
    if output.exists():
        raise FileExistsError('Preserve existing holdout result; do not run again')
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    verify_files(root, protocol)
    validate_lock(root/'data/lock.json', root/'data/manifest.jsonl')
    training = json.loads((root/'training-seed42/result.json').read_text(encoding='utf-8'))
    if training['selected'] != protocol['selected'] or training['config']['seed'] != protocol['seed']:
        raise ValueError('Selection differs from validation-frozen candidate')
    records = [json.loads(line) for line in (root/'data/manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    indices = [i for i,r in enumerate(records) if r['split'] == 'test']
    if len(indices) != 1440 or len({records[i]['prompt_group'] for i in indices}) != 240:
        raise ValueError('Unexpected test split')
    with np.load(root/'extraction/embeddings.npz', allow_pickle=False) as data:
        if data['pair_ids'].tolist() != [r['pair_id'] for r in records]:
            raise ValueError('Embedding ID mismatch')
        x = features(data['text'][indices], data['image'][indices])
        cosine = data['cosine'][indices]
    target = [records[i]['target'] for i in indices]
    groups = [records[i]['prompt_group'] for i in indices]
    # Reserve output before scoring so a completed or interrupted run is not overwritten.
    with output.open('x', encoding='utf-8') as stream:
        json.dump({'status':'RUNNING','protocol_sha256':sha256(protocol_path)}, stream)
    result = {'status':'PASS', 'scope':'frozen_seed42_test_holdout_not_release_acceptance',
              'protocol_sha256':sha256(protocol_path), 'selected':'mlp', 'seed':42,
              'count':len(indices), 'prompt_groups':240, 'models':{}}
    predictions = {}
    for kind in ('mean','ridge','mlp'):
        prediction = load_head(root/f'training-seed42/{kind}.pt').predict(x)
        result['models'][kind] = {**regression_metrics(target,prediction),
            'pairwise':pairwise_accuracy(target,prediction,groups,min_gap=protocol['min_label_gap'])}
        predictions[kind] = prediction.tolist()
    result['models']['clip_cosine'] = {'spearman':spearman(target,cosine),
        'mae':None,'rmse':None,'score_type':'raw_cosine_not_1_to_5',
        'pairwise':pairwise_accuracy(target,cosine,groups,min_gap=protocol['min_label_gap'])}
    result['predictions'] = {'pair_ids':[records[i]['pair_id'] for i in indices],
                             **predictions, 'cosine':cosine.tolist()}
    verify_files(root,protocol)
    output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='predictions'},indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    evaluate(args.root,args.protocol,args.output)
