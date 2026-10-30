"""Bounded train/validation experiment after baseline holdout was opened."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.data.acceptance import sha256,validate_lock
from src.models.interaction_features import interaction_features
from src.models.regression_head import save_head,load_head
from src.training.train_regression import fit_heads
from src.evaluation.metrics import pairwise_accuracy


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--features',nargs='+',choices=['cosine','interaction','hybrid'],default=['cosine','interaction'])
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    root=args.root
    manifest=root/'data/manifest.jsonl'
    validate_lock(root/'data/lock.json',manifest)
    extraction=json.loads((root/'extraction/result.json').read_text(encoding='utf-8'))
    embedding=root/'extraction/embeddings.npz'
    if sha256(embedding)!=extraction['embeddings_sha256'] or sha256(manifest)!=extraction['manifest_sha256']:
        raise ValueError('Input checksum mismatch')
    records=[json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines()]
    with np.load(embedding,allow_pickle=False) as data:
        if data['pair_ids'].tolist()!=[r['pair_id'] for r in records]:
            raise ValueError('Embedding order mismatch')
        # Access targets/features only for train and validation; test not evaluated.
        partitions={}
        for split in ('train','val'):
            ix=[i for i,r in enumerate(records) if r['split']==split]
            partitions[split]=(data['text'][ix],data['image'][ix],np.array([records[i]['target'] for i in ix]))
    groups=[r['prompt_group'] for r in records if r['split']=='val']
    protocol={'scope':'post_holdout_development_train_val_only', 'seeds':[42,43,44],
              'features':args.features,'alpha':1.0,'hidden':64,'epochs':100,
              'patience':15,'learning_rate':0.001,'manifest_sha256':sha256(manifest),
              'embeddings_sha256':sha256(embedding),'test':'NOT EVALUATED',
              'primary_seed':42,'promotion':'No automatic deployment; independent new set required'}
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    results={}
    for feature in protocol['features']:
        xtr=interaction_features(*partitions['train'][:2],feature)
        xval=interaction_features(*partitions['val'][:2],feature)
        for seed in protocol['seeds']:
            models,report=fit_heads(xtr,partitions['train'][2],xval,partitions['val'][2],seed=seed)
            folder=args.output/f'{feature}-seed{seed}'
            folder.mkdir()
            for kind,head in models.items():
                path=folder/f'{kind}.pt'
                before=head.predict(xval)
                report['metrics'][kind]['pairwise']=pairwise_accuracy(partitions['val'][2],before,groups)
                save_head(head,path)
                delta=float(np.max(np.abs(load_head(path).predict(xval)-before)))
                if delta>1e-5: raise ValueError('Reload mismatch')
                report.setdefault('checkpoints',{})[kind]={'sha256':sha256(path),'reload_delta':delta}
            report.update(feature_map=feature,seed=seed,scope=protocol['scope'])
            (folder/'result.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
            results[f'{feature}-seed{seed}']=report
            print(feature,seed,{k:round(v['mae'],5) for k,v in report['metrics'].items()},flush=True)
    aggregate={}
    for feature in protocol['features']:
        for kind in ('ridge','mlp'):
            values=[results[f'{feature}-seed{s}']['metrics'][kind]['mae'] for s in protocol['seeds']]
            aggregate[f'{feature}-{kind}']={'mae_mean':float(np.mean(values)),
                'mae_std_population':float(np.std(values)), 'mae_each':values}
    summary={'status':'PASS_EXECUTION_NOT_RELEASE','protocol':protocol,'aggregate':aggregate,
             'runs':{k:{'metrics':v['metrics'],'best_epoch':v['best_epoch']} for k,v in results.items()}}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(aggregate,indent=2),flush=True)


if __name__=='__main__': main()
