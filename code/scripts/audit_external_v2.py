"""Read-only intake audit; does not infer rating authenticity or run models."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from PIL import Image


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path('../data/image_eval_dataset_v2_20_anh')
    output = Path('../project_records/reports/results/G7_external_v2_intake_20261030.json')
    with (root/'prompts.csv').open(encoding='utf-8-sig', newline='') as f:
        prompts = list(csv.DictReader(f))
    rating_path = root/'ratings_simulated_3_students.csv'
    with rating_path.open(encoding='utf-8-sig', newline='') as f:
        ratings = list(csv.DictReader(f))
    errors, images = [], []
    ids = [Path(p['image_path']).name for p in prompts]
    if len(prompts) != 20 or len(set(ids)) != 20:
        errors.append('Expected 20 unique image IDs')
    if any(not p['prompt'].strip() for p in prompts):
        errors.append('Empty prompt')
    keys = [(r['image_id'], r['rater']) for r in ratings]
    if len(keys) != 60 or len(set(keys)) != 60:
        errors.append('Expected 60 unique image/rater keys')
    if set(r['image_id'] for r in ratings) != set(ids):
        errors.append('Ratings image ID mismatch')
    for image_id in ids:
        rows = [r for r in ratings if r['image_id'] == image_id]
        if {r['rater'] for r in rows} != {'Hoan', 'Huy', 'An'}:
            errors.append(f'Unexpected raters: {image_id}')
    for r in ratings:
        if r['score'] not in {'1', '2', '3', '4', '5'}:
            errors.append(f'Invalid score: {r["image_id"]}/{r["rater"]}')
    old_hashes = {digest(p) for p in Path('../data/external20/images/images').glob('*.png')}
    for p in prompts:
        path = (root/p['image_path']).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError('Image path outside intake folder')
        with Image.open(path) as im:
            im.load()
            width, height = im.size
            item = {'image_id': path.name, 'sha256': digest(path), 'width': width,
                    'height': height, 'format': im.format, 'bytes': path.stat().st_size}
        item['exact_duplicate_of_external20'] = item['sha256'] in old_hashes
        if width*height > 20_000_000 or item['bytes'] > 10*1024*1024:
            errors.append(f'Upload limit exceeded: {path.name}')
        images.append(item)
    if len(set(i['sha256'] for i in images)) != len(images):
        errors.append('Duplicate image bytes within new set')
    if any(i['exact_duplicate_of_external20'] for i in images):
        errors.append('Exact image bytes overlap old set')
    report = {'status': 'PASS_STRUCTURE' if not errors else 'FAIL_STRUCTURE',
        'errors': errors, 'prompt_count': len(prompts), 'rating_count': len(ratings),
        'unique_prompt_count': len({p['prompt'].strip() for p in prompts}),
        'score_distribution': dict(Counter(r['score'] for r in ratings)),
        'all_raters_same_score_images': sum(len({r['score'] for r in ratings if r['image_id']==i})==1 for i in ids),
        'files': {p.name: digest(p) for p in (root/'prompts.csv', rating_path)},
        'images': images, 'rating_provenance': 'PENDING_USER_CONFIRMATION',
        'inference': 'NOT RUN', 'acceptance': 'BLOCKED_PENDING_RATING_PROVENANCE',
        'limitations': ['Byte hashes do not rule out visually similar or transformed duplicates',
                       '20 unique prompts with one image each cannot measure within-prompt ranking',
                       'Scores only 4/5 cannot establish rejection of strongly mismatched pairs',
                       'Matching rater scores do not prove either authenticity or fabrication']}
    with output.open('x', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != 'images'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
