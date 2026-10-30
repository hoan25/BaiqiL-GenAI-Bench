import json
from pathlib import Path
import pytest
from src.inference.ensemble import EnsemblePredictor


def test_trial_requires_opt_in():
    with pytest.raises(ValueError, match='opt-in'):
        EnsemblePredictor('unused','unused','unused')


def test_trial_rejects_changed_evidence(tmp_path):
    config=tmp_path/'configs'/'trial.json'
    config.parent.mkdir()
    (tmp_path/'report.json').write_text('{}')
    config.write_text(json.dumps({'report':'report.json','report_sha256':'wrong'}))
    with pytest.raises(ValueError, match='checksum'):
        EnsemblePredictor('unused','unused',config,True)
