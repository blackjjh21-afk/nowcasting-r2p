from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cnn_readout.model import CNN
from cnn_readout.loss import importance_corrected_mse, normalize_target, millimetres_from_normalized_prediction
from cnn_readout import workflow


def test_cnn_convolutional_parameters_query_independence_and_gradients():
    torch.manual_seed(17)
    model = CNN().eval()
    assert any(isinstance(m, torch.nn.Conv2d) for m in model.modules())
    torch.nn.init.normal_(model.output[-1].weight, std=0.01)
    x, aux = torch.randn(4, 6, 3, 3), torch.randn(4, 39)
    whole = model(x, aux)
    separate = torch.cat([model(x[i:i+1], aux[i:i+1]) for i in range(4)])
    torch.testing.assert_close(whole, separate, rtol=1e-5, atol=1e-7)
    assert whole.shape == (4,) and (whole >= 0).all()
    whole.sum().backward()
    assert torch.isfinite(model.projection[0].weight.grad).all()


def test_importance_correction_estimates_normalized_mse():
    target = torch.tensor([0., 2., 7., 15., 25.])
    prediction = torch.tensor([0., 0.01, 0.02, 0.03, 0.04])
    p = torch.tensor([0.5, 0.2, 0.15, 0.1, 0.05])
    q = torch.full((5,), 0.2)
    loss = importance_corrected_mse(prediction, normalize_target(target), target,
        natural_fractions=p, sampled_fractions=q)
    expected = ((prediction - target / 735.) ** 2 * p / q).mean()
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(millimetres_from_normalized_prediction(torch.tensor([-1., 0., 2.])), torch.tensor([0., 0., 1470.]))


def test_saved_cnn_checkpoint_round_trip_and_schema_rejects_other_models(tmp_path):
    model = CNN().eval()
    checkpoint = tmp_path / "cnn.pt"
    torch.save({"schema": workflow.CHECKPOINT_SCHEMA,
                "model_config": {"patch_steps": 6, "patch_size": 3, "aux_dim": 39},
                "state_dict": model.state_dict()}, checkpoint)
    loaded, _ = workflow.load_checkpoint(checkpoint, torch.device("cpu"))
    loaded.eval()
    x, aux = torch.randn(2, 6, 3, 3), torch.randn(2, 39)
    torch.testing.assert_close(loaded(x, aux), model(x, aux))
    torch.save({"schema": "unrelated_checkpoint", "state_dict": {}}, checkpoint)
    with pytest.raises(RuntimeError, match="unsupported CNN"):
        workflow.load_checkpoint(checkpoint, torch.device("cpu"))
