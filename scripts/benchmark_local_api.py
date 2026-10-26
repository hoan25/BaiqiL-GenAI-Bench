"""Measure real local HTTP inference on synthetic inputs; not quality or an SLA."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
from pathlib import Path
import platform
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image
from backend.app.main import create_app
from src.inference.predictor import Predictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--training', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    predictor = Predictor(args.model, args.training)
    app = create_app(predictor)
    stream = BytesIO()
    Image.new('RGB', (224, 224), 'red').save(stream, format='PNG')
    payload = stream.getvalue()
    def request(count):
        with TestClient(app) as client:
            start = time.perf_counter()
            response = (client.post('/api/evaluate', data={'prompt': 'a red square'},
                        files={'image': ('red.png', payload, 'image/png')}) if count == 1 else
                        client.post('/api/compare', data={'prompt': 'a red square'},
                        files=[('images', ('red.png', payload, 'image/png'))] * count))
            elapsed = (time.perf_counter() - start) * 1000
        if response.status_code != 200:
            raise ValueError(response.text)
        values = response.json()['data']['items']
        assert len(values) == count and all(1 <= v['score'] <= 5 for v in values)
        return elapsed, values
    report = {'scope': 'synthetic_image_real_cpu_model_http_inprocess_not_quality_or_network_SLA',
              'model_version': predictor.version, 'platform': platform.platform(),
              'processor': platform.processor(), 'python': platform.python_version(),
              'memory_peak': 'NOT MEASURED', 'runs': {}}
    for count in (1, 6):
        request(count)
        samples = [request(count)[0] for _ in range(20)]
        report['runs'][str(count)] = {'requests': 20, 'warmups': 1, 'milliseconds': samples,
                                     'p50_ms': float(np.percentile(samples, 50)),
                                     'p95_ms': float(np.percentile(samples, 95))}
        print(f'batch {count}: {report["runs"][str(count)]["p95_ms"]:.1f} ms p95', flush=True)
    baseline = request(1)[1][0]['score']
    with ThreadPoolExecutor(max_workers=5) as pool:
        concurrent = list(pool.map(request, [1] * 5))
    delta = max(abs(items[0]['score'] - baseline) for _, items in concurrent)
    assert delta < 1e-5
    report.update(status='PASS', concurrent_requests=5, concurrent_max_score_delta=delta,
                  concurrency_scope='Shared predictor serialization only; session isolation covered by API tests')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('PASS')


if __name__ == '__main__':
    main()
