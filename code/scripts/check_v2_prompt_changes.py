"""Frozen candidate sensitivity check; no fitted labels or human-rating metrics."""
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from src.data.acceptance import sha256
from src.inference.predictor import Predictor
from src.models.regression_head import features, load_head
from src.models.interaction_features import interaction_features

# One explicit detail change per image, fixed before inference.
EDITS = [
    ('color','yellow ceramic teapot','red ceramic teapot'),
    ('color','green backpack','red backpack'),
    ('color','purple umbrella','yellow umbrella'),
    ('color','white rabbit','black rabbit'),
    ('count','Exactly two oranges','Exactly three oranges'),
    ('count','Exactly three lit candles','Exactly two lit candles'),
    ('count','Exactly four closed books','Exactly six closed books'),
    ('count','Exactly five yellow rubber ducks','Exactly three yellow rubber ducks'),
    ('position','to the left of','to the right of'),
    ('position','to the right of','to the left of'),
    ('position','underneath a wooden bench','on top of a wooden bench'),
    ('position','inside a clear glass bowl','outside a clear glass bowl'),
    ('position','on top of a blue cube','underneath a blue cube'),
    ('color','white ceramic vase','red ceramic vase'),
    ('state','closed blue umbrella','open blue umbrella'),
    ('color','red cup','blue cup'),
    ('color','cat wears a red collar','cat wears a green collar'),
    ('object','one lemon','one strawberry'),
    ('text','"BAKERY"','"PHARMACY"'),
    ('color','white face','green face'),
]


def main():
    root=Path('../data/image_eval_dataset_v2_20_anh')
    output=Path('../project_records/reports/results/G7_v2_prompt_changes_20261030')
    output.mkdir(exist_ok=False)
    with (root/'prompts.csv').open(encoding='utf-8-sig') as f:
        samples=list(csv.DictReader(f))
    assert len(samples)==len(EDITS)==20
    selected=json.loads(Path('../project_records/reports/results/G7_synthetic_ensembles_20261029.json').read_text())['results']['0.25']['checkpoints']
    for cp in selected:
        cp['path'] = str(Path(__file__).resolve().parents[2] / cp['path'])
        assert sha256(cp['path'])==cp['sha256']
    cases=[]
    for i,(s,(kind,before,after)) in enumerate(zip(samples,EDITS)):
        assert Path(s['image_path']).name==f'{i+1:02}.png'
        assert s['prompt'].count(before)==1
        cases.append({'image_path':s['image_path'],'image_sha256':sha256(root/s['image_path']),
            'category':kind,'original':s['prompt'],'detail_changed':s['prompt'].replace(before,after),
            'unrelated':samples[(i+7)%20]['prompt']})
    protocol={'scope':'diagnostic_expected_order_not_human_ground_truth',
        'expectation':'Original score exceeds changed-prompt score; user states originals match images',
        'tie_tolerance':1e-5,'meaningful_drop':0.25,'candidate':selected,
        'baseline':'reviewed seed42 MLP','cases':cases,'training':'NOT RUN',
        'human_or_simulated_ratings':'NOT USED','holdout':'NOT EVALUATED'}
    (output/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    predictor=Predictor('../.cache/models/clip-vit-base-patch32/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268',
        '../.cache/reviewed-training-20261020T174855207187Z/training-seed42')
    heads=[load_head(cp['path']) for cp in selected]
    rows=[]
    for c in cases:
        with Image.open(root/c['image_path']) as im: image=im.convert('RGB')
        enc=predictor.encoder.encode([c[k] for k in ('original','detail_changed','unrelated')],[image]*3)
        old=predictor.head.predict(features(enc['text'],enc['image']))
        x=interaction_features(enc['text'],enc['image'],'hybrid')
        new=np.mean([h.predict(x) for h in heads],axis=0)
        assert sha256(root/c['image_path'])==c['image_sha256']
        rows.append({'image':c['image_path'],'category':c['category'],
            'baseline':old.tolist(),'ensemble25':new.tolist(),'cosine':enc['cosine'].tolist()})
    summary={}
    for name in ('baseline','ensemble25','cosine'):
        summary[name]={}
        for j,kind in ((1,'detail_changed'),(2,'unrelated')):
            deltas=np.array([r[name][0]-r[name][j] for r in rows])
            summary[name][kind]={'expected_order_count':int(np.sum(deltas>1e-5)),
                'ties':int(np.sum(np.abs(deltas)<=1e-5)), 'total':len(rows),
                'mean_drop':float(deltas.mean()),
                'drop_at_least_0_25':int(np.sum(deltas>=0.25)) if name!='cosine' else None}
    report={'scope':protocol['scope'],'score_order':['original','detail_changed','unrelated'],
        'summary':summary,'rows':rows,'protocol_sha256':sha256(output/'protocol.json'),
        'release':'NOT ACCEPTED; diagnostic only'}
    (output/'result.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    with (output/'scores.csv').open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f); w.writerow(['image','category','model','original','detail_changed','unrelated'])
        for r in rows:
            for name in ('baseline','ensemble25','cosine'):w.writerow([r['image'],r['category'],name,*r[name]])
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
