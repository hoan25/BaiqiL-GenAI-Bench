"""Equal-weight all-seed ensembles, development evidence only."""
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.data.acceptance import sha256
from src.models.interaction_features import interaction_features
from src.models.regression_head import load_head
from src.evaluation.metrics import regression_metrics, pairwise_accuracy


def main():
    root=Path('../.cache/reviewed-training-20261020T174855207187Z')
    experiment=Path('../.cache/synthetic-negatives-20261029')
    output=Path('../project_records/reports/results/G7_synthetic_ensembles_20261029.json')
    if output.exists(): raise FileExistsError(output)
    protocol=json.loads((experiment/'protocol.json').read_text())
    ensemble_protocol={'weights':[1/3]*3, 'seeds':[42,43,44],
        'fractions':protocol['fractions'], 'scope':'post_holdout_development',
        'gate':protocol['provisional_gate'],
        'note':'Individual seed gate remains separate; ensemble is a new candidate, no independent test'}
    with (experiment/'ensemble_protocol.json').open('x',encoding='utf-8') as f:
        json.dump(ensemble_protocol,f,indent=2)
    manifest=root/'data/manifest.jsonl'
    embedding=root/'extraction/embeddings.npz'
    assert sha256(manifest)==protocol['manifest_sha256']
    assert sha256(embedding)==protocol['embeddings_sha256']
    records=[json.loads(s) for s in manifest.read_text(encoding='utf-8').splitlines()]
    ix=[i for i,r in enumerate(records) if r['split']=='val']
    with np.load(embedding,allow_pickle=False) as data:
        assert data['pair_ids'].tolist()==[r['pair_id'] for r in records]
        x=interaction_features(data['text'][ix],data['image'][ix],'hybrid')
    y=[records[i]['target'] for i in ix]
    groups=[records[i]['prompt_group'] for i in ix]
    with open('../data/external20/ratings.csv',encoding='utf-8-sig') as f:
        ratings=list(csv.DictReader(f))
    with open('../data/external20/samples.csv',encoding='utf-8-sig') as f:
        samples=list(csv.DictReader(f))
    ey=[np.mean([float(r['score_1_to_5']) for r in ratings if r['sample_id']==s['sample_id']]) for s in samples]
    diagnostic_path=Path('../project_records/reports/results/G7_synthetic_diagnostics_20261029.json')
    diagnostic=json.loads(diagnostic_path.read_text())
    assert diagnostic['ratings_sha256']==sha256('../data/external20/ratings.csv')
    results={}
    for fraction in protocol['fractions']:
        vp, ep, dp, checkpoints=[],[],[],[]
        for seed in ensemble_protocol['seeds']:
            name=f'negative{fraction}-seed{seed}'
            folder=experiment/name
            run=json.loads((folder/'result.json').read_text())
            path=folder/'mlp.pt'
            assert sha256(path)==run['checkpoints']['mlp']['sha256']
            model=load_head(path)
            batch=model.predict(x)
            single=np.array([model.predict(row[None])[0] for row in x[:16]])
            delta=float(np.max(np.abs(single-batch[:16])))
            assert delta<1e-5
            vp.append(batch)
            d=diagnostic['candidate_results'][name+'-mlp']
            ep.append(d['external20_scores']); dp.append(d['diagnostic_scores'])
            checkpoints.append({'path':path.resolve().relative_to(Path(__file__).resolve().parents[2]).as_posix(),
                                'sha256':sha256(path),'batch_single_delta':delta})
        v=np.mean(vp,axis=0); e=np.mean(ep,axis=0); d=np.mean(dp,axis=0)
        vm=regression_metrics(y,v); vm['pairwise']=pairwise_accuracy(y,v,groups)
        em=regression_metrics(ey,e)
        checks=[bool(d[0]>d[1]),bool(d[3]>d[5]),bool(d[3]>d[7])]
        gate=ensemble_protocol['gate']
        accepted=(vm['mae']<=gate['validation_mae_max_each_seed'] and
            em['mae']<=gate['external20_seen_mae_max_each_seed'] and all(checks))
        results[str(fraction)]={'validation':vm,'external20_seen':em,
            'diagnostic_scores':d.tolist(),'external20_scores':e.tolist(),
            'ordering_checks':checks,'provisional_development_gate':bool(accepted),
            'checkpoints':checkpoints, 'member_external20_mae':[regression_metrics(ey,p)['mae'] for p in ep]}
    report={'protocol':ensemble_protocol,'diagnostics_sha256':sha256(diagnostic_path),
        'results':results,'holdout':'NOT EVALUATED','release':'NOT ACCEPTED'}
    with output.open('x',encoding='utf-8') as f: json.dump(report,f,indent=2)
    for fraction,r in results.items():
        print(fraction,r['validation']['mae'],r['external20_seen'],r['provisional_development_gate'])


if __name__=='__main__': main()
