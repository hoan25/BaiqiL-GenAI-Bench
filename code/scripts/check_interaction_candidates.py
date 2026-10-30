"""Check development candidates on seen diagnostic cases, not independent test."""
import argparse
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from src.inference.predictor import Predictor
from src.models.regression_head import load_head
from src.models.interaction_features import interaction_features
from src.evaluation.metrics import regression_metrics
from src.data.acceptance import sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiments',nargs='+',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    root=Path('../data/external20')
    with (root/'samples.csv').open(encoding='utf-8-sig') as f:samples=list(csv.DictReader(f))
    with (root/'ratings.csv').open(encoding='utf-8-sig') as f:ratings=list(csv.DictReader(f))
    target=[np.mean([int(r['score_1_to_5']) for r in ratings if r['sample_id']==s['sample_id']]) for s in samples]
    predictor=Predictor('../.cache/models/clip-vit-base-patch32/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268',
                        '../.cache/reviewed-training-20261020T174855207187Z/training-seed42')
    cases=[(s['evaluation_prompt_en'],root/s['image_path']) for s in samples]
    prompts=['A red apple on a white plate.','A red car parked in front of a white house.',
             'A red bicycle parked in front of a white house.','An astronaut walking on the moon.']
    cases += [(prompt,root/f'images/images/{id}.png') for prompt in prompts for id in ('01','20')]
    enc=[]
    for start in range(0,len(cases),4):
        batch=cases[start:start+4]
        images=[]
        for _,file in batch:
            with Image.open(file) as im:images.append(im.convert('RGB'))
        enc.append(predictor.encoder.encode([s for s,_ in batch],images))
    text=np.concatenate([r['text'] for r in enc]);image=np.concatenate([r['image'] for r in enc])
    result={'scope':'seen_external20_and_prompt_diagnostics_not_independent_validation',
            'test_holdout':'NOT EVALUATED','candidate_results':{},
            'diagnostic_order':[{'prompt':prompt,'image':file.name} for prompt,file in cases[20:]],
            'ratings_sha256':sha256(root/'ratings.csv')}
    for experiment in a.experiments:
        for folder in sorted(experiment.glob('*-seed*')):
            run=json.loads((folder/'result.json').read_text())
            x=interaction_features(text,image,run['feature_map'])
            for kind in ('ridge','mlp'):
                path=folder/f'{kind}.pt'
                if sha256(path)!=run['checkpoints'][kind]['sha256']:raise ValueError('Checkpoint changed')
                scores=load_head(path).predict(x)
                result['candidate_results'][f'{folder.name}-{kind}']={
                    'external20':regression_metrics(target,scores[:20]),
                    'external20_scores':scores[:20].tolist(),'diagnostic_scores':scores[20:].tolist(),
                    'validation':run['metrics'][kind],
                    'apple_beats_car_for_apple_prompt':bool(scores[20]>scores[21]),
                    'car_correct_beats_bicycle_prompt':bool(scores[23]>scores[25]),
                    'car_correct_beats_astronaut_prompt':bool(scores[23]>scores[27])}
    with a.output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    for name,r in result['candidate_results'].items():
        print(name,'val',round(r['validation']['mae'],4),'external_seen',round(r['external20']['mae'],4),
              'diagnostics',r['apple_beats_car_for_apple_prompt'],r['car_correct_beats_bicycle_prompt'],r['car_correct_beats_astronaut_prompt'],flush=True)


if __name__=='__main__':main()
