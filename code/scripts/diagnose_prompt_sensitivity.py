"""Post-demo diagnostic on two known external images, not a new benchmark."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from src.data.acceptance import sha256
from src.inference.predictor import Predictor
from src.models.regression_head import features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--training', required=True)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    predictor = Predictor(args.model, args.training)
    prompts = [
        'A red apple on a white plate.',
        'A red car parked in front of a white house.',
        'A red bicycle parked in front of a white house.',
        'An astronaut walking on the moon.',
    ]
    paths = [args.images/'01.png', args.images/'20.png']
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert('RGB'))
    hashes = {p.name:sha256(p) for p in paths}
    rows, max_delta = [], 0.0
    for prompt in prompts:
        public = predictor.predict(prompt, images)
        encoded = predictor.encoder.encode([prompt]*2, images)
        direct = predictor.head.predict(features(encoded['text'], encoded['image']))
        delta = float(np.max(np.abs(direct - [r['score'] for r in public])))
        max_delta = max(max_delta, delta)
        for i, path in enumerate(paths):
            rows.append({'image':path.name, 'prompt':prompt, **public[i]})
    if max_delta > 1e-5:
        raise ValueError('Public predictor and direct head disagree')
    if hashes != {p.name:sha256(p) for p in paths}:
        raise ValueError('Input images changed')
    result = {'status':'PASS_EXECUTION_NOT_MODEL_QUALITY',
              'scope':'post_demo_diagnostic_seen_external_images_no_fit_no_holdout',
              'model_version':predictor.version, 'image_sha256':hashes,
              'training_report_sha256':sha256(Path(args.training)/'result.json'),
              'public_vs_direct_max_delta':max_delta, 'rows':rows,
              'score_range_across_prompts':{
                  path.name:float(np.ptp([r['score'] for r in rows if r['image']==path.name]))
                  for path in paths},
              'limitations':['Two selected images and four prompts only',
                             'No new human ground truth or quality accuracy metric',
                             'Does not establish the training-level cause']}
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
