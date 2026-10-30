import numpy as np
import pytest
import torch

from src.models.regression_head import features, save_head, load_head
from src.training.train_regression import fit_heads, train_from_embeddings


def fixture():
    x = np.linspace(-1, 1, 32, dtype=np.float32)[:, None]
    return np.concatenate([x, np.ones_like(x)], axis=1), 3 + x[:, 0]


def test_ridge_reference_scaler_train_only_and_reload(tmp_path):
    x, y = fixture()
    val = x + np.array([10, 0], dtype=np.float32)
    models, report = fit_heads(x, y, val, y, epochs=3)
    ridge = models["ridge"]
    np.testing.assert_allclose(ridge.center.numpy(), x.mean(axis=0), atol=1e-7)
    assert ridge.scale.numpy()[1] == 1
    z = (x[:, 0] - x[:, 0].mean()) / x[:, 0].std()
    expected = np.dot(z, y - y.mean()) / (np.dot(z, z) + 1)
    assert ridge.network.weight.detach().numpy()[0, 0] == pytest.approx(expected, abs=1e-6)
    for name, model in models.items():
        path = tmp_path / f"{name}.pt"
        save_head(model, path)
        np.testing.assert_allclose(load_head(path).predict(x), model.predict(x), atol=1e-6)
        with pytest.raises(FileExistsError):
            save_head(model, path)
        assert np.all((model.predict(val) >= 1) & (model.predict(val) <= 5))


def test_mlp_overfits_small_fixture_and_best_epoch_restored():
    x, y = fixture()
    models, report = fit_heads(x, y, x, y, epochs=250, patience=40, learning_rate=0.03, hidden=16)
    history = report["history"]
    assert min(h["train_mse_before_step"] for h in history) < history[0]["train_mse_before_step"] * 0.1
    assert report["metrics"]["mlp"]["mae"] < 0.08
    best = min(history, key=lambda h: h["val_mae"])
    assert report["best_epoch"] == best["epoch"]
    assert report["metrics"]["mlp"]["mae"] == pytest.approx(best["val_mae"])


def test_seed_reproducibility_and_early_stop():
    x, y = fixture()
    _, first = fit_heads(x, y, x, 6 - y, epochs=50, patience=2, learning_rate=0.02)
    _, second = fit_heads(x, y, x, 6 - y, epochs=50, patience=2, learning_rate=0.02)
    assert first == second
    assert first["epochs_run"] < 50
    assert first["metrics"][first["selected"]]["mae"] == min(m["mae"] for m in first["metrics"].values())


def test_embedding_alignment_and_no_test_target_or_features():
    records = [dict(pair_id=str(i), prompt=f"prompt {i}", split=split, **({"target": 2 + i} if split != "test" else {}))
               for i, split in enumerate(["train", "val", "test"])]
    arrays = {"pair_ids": np.array(["0", "1", "2"]), "text": np.array([[0.], [1.], [np.nan]]),
              "image": np.array([[0.], [1.], [np.nan]]), "cosine": np.array([0., 0.5, np.nan])}
    _, report, _ = train_from_embeddings(records, arrays, epochs=2)
    assert report["train_count"] == report["val_count"] == 1
    arrays["pair_ids"] = arrays["pair_ids"][::-1]
    with pytest.raises(ValueError, match="pair IDs"):
        train_from_embeddings(records, arrays, epochs=2)


@pytest.mark.parametrize("kwargs", [{"alpha": 0}, {"epochs": 0}, {"patience": 0}, {"learning_rate": float('nan')}])
def test_invalid_config(kwargs):
    x, y = fixture()
    with pytest.raises(ValueError):
        fit_heads(x, y, x, y, **kwargs)


def test_feature_contract_and_nonfinite_inputs():
    assert features([[1, 2]], [[3, 4]]).tolist() == [[1, 2, 3, 4]]
    with pytest.raises(ValueError):
        features([[np.nan]], [[1]])
    x, y = fixture()
    y[0] = 6
    with pytest.raises(ValueError, match="targets"):
        fit_heads(x, y, x, y)


def test_cuda_request_does_not_silently_fall_back(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    x, y = fixture()
    with pytest.raises(ValueError, match='unavailable'):
        fit_heads(x, y, x, y, device='cuda')
    with pytest.raises(ValueError, match='device'):
        fit_heads(x, y, x, y, device='gpu')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Real CUDA hardware required')
def test_real_cuda_training_and_cpu_export(tmp_path):
    x, y = fixture()
    models, report = fit_heads(x, y, x, y, device='cuda', epochs=120, patience=30,
                              hidden=16, learning_rate=0.03)
    assert report['training_device'] == 'cuda'
    assert report['device_to_cpu_max_delta'] <= 1e-4
    assert report['metrics']['mlp']['mae'] < 0.15
    cpu, _ = fit_heads(x, y, x, y, epochs=1)
    np.testing.assert_allclose(models['ridge'].predict(x), cpu['ridge'].predict(x), atol=1e-5)
    for kind, model in models.items():
        assert model.center.device.type == 'cpu'
        path = tmp_path/f'{kind}.pt'
        save_head(model, path)
        np.testing.assert_allclose(load_head(path).predict(x), model.predict(x), atol=1e-6)
