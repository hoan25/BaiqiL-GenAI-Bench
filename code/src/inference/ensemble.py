"""Opt-in local ensemble trial; baseline receipts and encoder checks remain enforced."""
import json
from pathlib import Path
import numpy as np
from src.data.acceptance import sha256
from src.inference.predictor import Predictor
from src.models.regression_head import load_head
from src.models.interaction_features import interaction_features


class EnsemblePredictor(Predictor):
    def __init__(self, model_path, training_path, config_path, allow_development=False):
        if not allow_development:
            raise ValueError('Ensemble trial requires explicit development opt-in')
        config_path = Path(config_path).resolve()
        config = json.loads(config_path.read_text(encoding='utf-8'))
        root = config_path.parent.parent
        if root.name == "code":
            root = root.parent
        report_path = (root/config['report']).resolve()
        if not report_path.is_relative_to(root) or sha256(report_path) != config['report_sha256']:
            raise ValueError('Ensemble evidence checksum/path mismatch')
        evidence = json.loads(report_path.read_text(encoding='utf-8'))
        selected = evidence['results'][config['candidate']]
        if len(selected['checkpoints']) != 3:
            raise ValueError('Expected three ensemble members')
        heads = []
        for cp in selected['checkpoints']:
            path = (root/cp['path']).resolve()
            if not path.is_relative_to(root) or sha256(path) != cp['sha256']:
                raise ValueError('Ensemble checkpoint checksum/path mismatch')
            head = load_head(path)
            if head.config['kind'] != 'mlp':
                raise ValueError('Expected MLP ensemble member')
            heads.append(head)
        super().__init__(model_path, training_path)
        expected = 4*self.encoder.model.config.projection_dim+1
        if any(h.config['dimension'] != expected for h in heads):
            raise ValueError('Ensemble feature dimension mismatch')
        self.heads = heads
        self.version = 'ensemble25-' + sha256(config_path)[:12]
        self.scope = 'experimental_ensemble_validation'
        self.warnings = ['Mô hình ensemble thử nghiệm; còn hạn chế đếm và trạng thái. Chưa nghiệm thu release.']
        self.metrics['models'] = {**self.metrics['models'], 'ensemble25': selected['validation']}
        self.metrics.update(scope=self.scope, selected='ensemble25')

    def predict(self, prompt, images):
        with self.lock:
            encoded = self.encoder.encode([prompt]*len(images), images)
            x = interaction_features(encoded['text'], encoded['image'], 'hybrid')
            scores = np.mean([head.predict(x) for head in self.heads], axis=0)
        return [{'score':float(s),'cosine':float(c)} for s,c in zip(scores,encoded['cosine'])]
