"""Small CPU-only shape, physics-unit, gradient and masked-loss tests."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nowcastnet_hsr.model import ModelConfig, NowcastNet, Discriminator, DiscriminatorLBlock, warp
from nowcastnet_hsr.losses import (
    evolution_loss, generator_loss, discriminator_loss, weighted_l1,
    pool_regularization, motion_regularization,
)

torch.set_num_threads(1)


def small_model():
    torch.manual_seed(17)
    return NowcastNet(ModelConfig(input_steps=3, output_steps=2, base_channels=4, noise_channels=4))


def history():
    torch.manual_seed(19)
    return torch.rand(1, 3, 64, 64) * 8


def test_complete_model_shapes_and_disjoint_parameter_groups():
    model = small_model()
    field = history()
    evolution = model.evolve(field)
    assert evolution["prediction"].shape == (1, 2, 64, 64)
    assert evolution["advected"].shape == (1, 2, 64, 64)
    assert evolution["motion"].shape == (1, 2, 2, 64, 64)
    output = model.generate(field, evolution)
    assert output.shape == (1, 2, 64, 64)
    assert torch.isfinite(output).all()
    e = {id(p) for p in model.evolution_parameters()}
    g = {id(p) for p in model.generation_parameters()}
    assert e and g and not e.intersection(g)
    assert e | g == {id(p) for p in model.parameters()}


def test_official_noise_spectral_state_updates_in_eval_and_is_restorable():
    model = small_model()
    field = history()
    generator = torch.Generator().manual_seed(9)
    noise = model.sample_noise(1, 64, 64, "cpu", generator)
    # Legacy official SN starts with random power vectors. Warm its statistics
    # before asserting that noise is numerically visible in an untrained model.
    with torch.no_grad():
        for _ in range(3):
            model(field, noise)
    model.eval()
    state = deepcopy(model.state_dict())
    with torch.no_grad():
        first = model(field, noise)
        assert not torch.equal(state["noise_projector.first.module.weight_u"],
                               model.noise_projector.first.module.weight_u)
        model.load_state_dict(state)
        repeated = model(field, noise)
        model.load_state_dict(state)
        different = model(field, model.sample_noise(1, 64, 64, "cpu", generator))
    torch.testing.assert_close(first, repeated, rtol=0, atol=0)
    assert not torch.equal(first, different)


def test_non_square_general_noise_projection():
    model = small_model().eval()
    field = torch.rand(1, 3, 64, 96)
    with torch.no_grad():
        result = model(field)
    assert result.shape == (1, 2, 64, 96)


def test_requested_seven_input_eighteen_output_contract():
    model = NowcastNet(ModelConfig(base_channels=4, noise_channels=4)).eval()
    field = torch.rand(1, 7, 64, 64)
    with torch.no_grad():
        result = model(field)
    assert result.shape == (1, 18, 64, 64)
    assert torch.isfinite(result).all()


def test_evolution_motion_gets_gradients_through_bilinear_auxiliary():
    model = small_model()
    field = history()
    evolution = model.evolve(field)
    evolution["motion"].retain_grad()
    target = field[:, -1:].expand(-1, 2, -1, -1) * .8
    losses = evolution_loss(evolution["prediction"], target, evolution["motion"],
                            torch.ones_like(target, dtype=torch.bool), advected=evolution["advected"])
    losses["total"].backward()
    assert torch.isfinite(losses["total"])
    assert evolution["motion"].grad.abs().sum() > 0
    assert model.evolution.motion_head.weight.grad.abs().sum() > 0
    assert model.evolution.gamma.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.generation_parameters())


def test_generation_backpropagation_and_decoupled_evolution():
    model = small_model()
    disc = Discriminator(in_steps=3, out_steps=2, base_channels=4)
    field = history()
    evolution = model.evolve(field)
    samples = torch.stack([model.generate(field, evolution) for _ in range(4)])
    logits = torch.stack([disc(field, item) for item in samples])
    target = torch.rand_like(samples[0]) * 6
    losses = generator_loss(samples, target, logits, torch.ones_like(target, dtype=torch.bool))
    losses["total"].backward()
    assert all(torch.isfinite(value) for value in losses.values())
    assert model.decoder.out.weight.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.noise_projector.parameters())
    assert all(p.grad is None for p in model.evolution_parameters())


def test_discriminator_trainable_conditional_logits():
    disc = Discriminator(in_steps=3, out_steps=2, base_channels=4)
    field = history()
    real, fake = torch.rand(1, 2, 64, 64), torch.zeros(1, 2, 64, 64)
    logits = disc(field, real)
    assert logits.shape == (1, 1, 4, 4)
    losses = discriminator_loss(logits, disc(field, fake))
    losses["total"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in disc.parameters())


@pytest.mark.parametrize("in_steps,out_steps,channels,temporal_lengths", [
    (4, 20, 188, (21, 5)),  # Extended Data Fig. 1c, not released 9-frame history.
    (9, 20, 248, (26, 10)),  # Same reconstructed discriminator with 9-frame history.
    (7, 18, 216, (22, 8)),
])
def test_discriminator_figure_release_length_and_kma_front_shapes(in_steps, out_steps, channels, temporal_lengths):
    disc = Discriminator(in_steps=in_steps, out_steps=out_steps, base_channels=32)
    past = torch.randn(1, in_steps, 64, 64)
    future = torch.randn(1, out_steps, 64, 64)
    captured = []
    handles = [layer.register_forward_hook(lambda _, args, output: captured.append(output.shape))
               for layer in disc.temporal]
    try:
        with torch.no_grad():
            front = disc.front_features(past, future)
            logits = disc.head(disc.blocks(front))
    finally:
        for handle in handles:
            handle.remove()
    assert captured == [(1, 4, temporal_lengths[0], 32, 32),
                        (1, 8, temporal_lengths[1], 32, 32)]
    assert front.shape == (1, channels, 32, 32)
    assert disc.front_channels == channels
    assert logits.shape == (1, 1, 4, 4)
    assert disc.spatial_front.kernel_size == (9, 9)
    assert disc.spatial_front.stride == (2, 2)
    assert disc.spatial_front.padding == (4, 4)
    assert [layer.kernel_size for layer in disc.temporal] == [(4, 9, 9), (out_steps, 9, 9)]
    assert all(layer.stride == (1, 2, 2) and layer.padding == (0, 4, 4) for layer in disc.temporal)


def test_discriminator_time_flatten_preserves_positions_and_channel_order():
    features = torch.arange(2 * 3 * 2 * 2.).reshape(1, 2, 3, 2, 2)
    flattened = Discriminator.flatten_time(features)
    assert flattened.shape == (1, 6, 2, 2)
    for channel in range(2):
        for time in range(3):
            torch.testing.assert_close(flattened[:, channel*3+time], features[:, channel, time],
                                       atol=0, rtol=0)
    # Opposite temporal order has identical temporal means, but must NOT have
    # the same discriminator front representation.
    reversed_features = features.flip(2)
    torch.testing.assert_close(features.mean(2), reversed_features.mean(2))
    assert not torch.equal(flattened, Discriminator.flatten_time(reversed_features))
    with pytest.raises(ValueError, match="Temporal features"):
        Discriminator.flatten_time(torch.zeros(1, 2, 3, 4))


def test_discriminator_l_blocks_bn_branches_spectral_norm_and_downsampling():
    disc = Discriminator(in_steps=3, out_steps=2, base_channels=4)
    assert len(disc.blocks) == 4
    assert [block.main[-1].out_channels for block in disc.blocks] == [16, 32, 64, 64]
    assert [block.main[-1].stride for block in disc.blocks] == [(2, 2), (2, 2), (2, 2), (1, 1)]
    for block in disc.blocks:
        assert isinstance(block, DiscriminatorLBlock)
        assert [type(layer) for layer in block.skip] == [torch.nn.BatchNorm2d, torch.nn.Conv2d]
        assert [type(layer) for layer in block.main] == [torch.nn.BatchNorm2d, torch.nn.ReLU, torch.nn.Conv2d]
        assert block.skip[-1].stride == block.main[-1].stride
        assert block.skip[-1].kernel_size == block.main[-1].kernel_size == (3, 3)
        assert block.skip[-1].padding == block.main[-1].padding == (1, 1)
    assert [type(layer) for layer in disc.head] == [torch.nn.BatchNorm2d, torch.nn.LeakyReLU, torch.nn.Conv2d]
    assert disc.head[1].negative_slope == .2
    assert disc.head[-1].kernel_size == (3, 3)
    for layer in disc.modules():
        if isinstance(layer, (torch.nn.Conv2d, torch.nn.Conv3d)):
            assert hasattr(layer, "weight_orig")
            assert hasattr(layer, "weight_u") and hasattr(layer, "weight_v")


def test_only_discriminator_bn_disables_and_removes_running_statistics():
    disc = Discriminator(in_steps=3, out_steps=2, base_channels=4)
    batch_norms = [module for module in disc.modules() if isinstance(module, torch.nn.BatchNorm2d)]
    assert len(batch_norms) == 9  # Two per L-block plus the final head.
    for module in batch_norms:
        assert not module.track_running_stats
        assert module.running_mean is None and module.running_var is None
        assert module.num_batches_tracked is None
        assert not list(module.named_buffers())
        assert set(module.state_dict()) == {"weight", "bias"}
    assert not any(key.endswith(("running_mean", "running_var", "num_batches_tracked"))
                   for key in disc.state_dict())
    # This is deliberately D-only; released evolution/generation BN remains
    # unchanged, including its inference-time running-statistics behavior.
    model = small_model()
    for subtree in (model.evolution, model.encoder):
        original_norms = [module for module in subtree.modules() if isinstance(module, torch.nn.BatchNorm2d)]
        assert original_norms
        assert all(module.track_running_stats and module.running_mean is not None
                   and module.running_var is not None and module.num_batches_tracked is not None
                   for module in original_norms)


def test_discriminator_bn_mode_keeps_outputs_and_input_gradients_identical():
    torch.manual_seed(123)
    disc = Discriminator(in_steps=3, out_steps=2, base_channels=4)
    past, future = torch.rand(2, 3, 64, 64) * 10, torch.rand(2, 2, 64, 64) * 20
    with torch.no_grad():
        for _ in range(3):
            disc(past, future)  # Prepare legacy SN vectors; no optimizer updates.
    disc.eval().requires_grad_(False)
    frozen_state = deepcopy(disc.state_dict())
    batch_norms = [module for module in disc.modules() if isinstance(module, torch.nn.BatchNorm2d)]
    for module in batch_norms:
        source = torch.randn(2, module.num_features, 4, 4)
        weights = torch.randn_like(source)
        outcomes = []
        for mode in (True, False):
            module.train(mode)
            value = source.clone().requires_grad_()
            output = module(value)
            gradient, = torch.autograd.grad((output * weights).sum(), value)
            outcomes.append((output.detach(), gradient))
        for trained, evaluated in zip(*outcomes):
            torch.testing.assert_close(trained, evaluated, atol=0, rtol=0)
    # Toggle ONLY BN, leaving every convolution eval so SN power iteration is
    # frozen. Thus differences cannot be hidden by changing spectral weights.
    outcomes = []
    weights = torch.randn(2, 1, 4, 4)
    for mode in (True, False):
        for module in batch_norms:
            module.train(mode)
        value = future.clone().requires_grad_()
        logits = disc(past, value)
        gradient, = torch.autograd.grad((logits * weights).sum(), value)
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
        outcomes.append((logits.detach(), gradient))
    for trained, evaluated in zip(*outcomes):
        torch.testing.assert_close(trained, evaluated, atol=0, rtol=0)
    for key, value in disc.state_dict().items():
        torch.testing.assert_close(value, frozen_state[key], atol=0, rtol=0)


def test_discriminator_raw_physical_input_and_tiny_temporal_geometry():
    disc = Discriminator(in_steps=3, out_steps=2, base_channels=4)
    past = torch.full((1, 3, 64, 64), 128.)
    future = torch.full((1, 2, 64, 64), 64.)
    captured = {}
    handle = disc.spatial_front.register_forward_pre_hook(
        lambda _, args: captured.update(value=args[0].detach().clone()))
    try:
        with torch.no_grad():
            logits = disc(past, future)
    finally:
        handle.remove()
    assert disc.input_scale == 1.0
    torch.testing.assert_close(captured["value"], torch.cat((past, future), dim=1), atol=0, rtol=0)
    assert torch.isfinite(logits).all()
    shortest = Discriminator(in_steps=1, out_steps=1, base_channels=4)
    assert shortest.temporal[0].kernel_size == (2, 9, 9)
    with torch.no_grad():
        assert shortest(past[:, :1], future[:, :1]).shape == (1, 1, 4, 4)
    with pytest.raises(ValueError, match="input scale"):
        Discriminator(input_scale=float("nan"))
    with pytest.raises(ValueError, match="integer"):
        Discriminator(in_steps=0)


def test_no_upper_cap_and_physical_rate_scale():
    model = small_model().eval()
    with torch.no_grad():
        model.evolution.motion_head.weight.zero_()
        model.evolution.motion_head.bias.zero_()
        model.decoder.out.weight.zero_()
        model.decoder.out.bias.fill_(300)
        field = torch.full((1, 3, 64, 64), 250.)
        evolution = model.evolve(field)
        generated = model.generate(field, evolution)
    torch.testing.assert_close(evolution["prediction"], torch.full_like(evolution["prediction"], 250.))
    torch.testing.assert_close(generated, torch.full_like(generated, 300.))


def test_official_raw_history_and_evolved_only_condition_scaling():
    model = small_model().eval()
    captured = {}
    handles = [model.evolution.register_forward_pre_hook(lambda _, args: captured.update(evolution=args[0].clone())),
               model.encoder.register_forward_pre_hook(lambda _, args: captured.update(encoder=args[0].clone()))]
    field = history()
    try:
        with torch.no_grad():
            evolved = model.evolve(field)
            model.generate(field, evolved)
    finally:
        for handle in handles:
            handle.remove()
    torch.testing.assert_close(captured["evolution"], field, atol=0, rtol=0)
    torch.testing.assert_close(captured["encoder"][:, :3], field, atol=0, rtol=0)
    torch.testing.assert_close(captured["encoder"][:, 3:], evolved["prediction"] / 128, atol=0, rtol=0)


def test_warp_departure_direction_and_identity():
    field = torch.arange(16.).reshape(1, 1, 4, 4)
    motion = torch.zeros(1, 2, 4, 4)
    torch.testing.assert_close(warp(field, motion, mode="nearest"), field)
    motion[:, 0] = 1
    moved = warp(field, motion, mode="nearest")
    torch.testing.assert_close(moved[..., :-1], field[..., 1:])
    torch.testing.assert_close(moved[..., -1], field[..., -1])


def test_masked_losses_ignore_missing_labels_not_zero_fill_truth():
    target = torch.ones(1, 2, 8, 8)
    target[..., 0, 0] = float("nan")
    predicted = torch.ones_like(target, requires_grad=True)
    loss = weighted_l1(predicted, target)
    assert loss.item() == 0
    loss.backward()
    assert predicted.grad[..., 0, 0].eq(0).all()
    mask = torch.ones_like(target, dtype=torch.bool)
    mask[..., 1, 1] = False
    target[..., 1, 1] = 500
    assert weighted_l1(predicted, target, mask).item() == 0
    with pytest.raises(ValueError, match="No valid"):
        weighted_l1(predicted, target, torch.zeros_like(mask))


def test_loss_weight_cap_is_not_rain_rate_cap():
    value = weighted_l1(torch.zeros(1, 1, 1, 1), torch.full((1, 1, 1, 1), 300.))
    assert value.item() == 24 * 300


def test_pool_each_member_before_ensemble_mean():
    samples = torch.zeros(2, 1, 1, 5, 5)
    samples[0, 0, 0, 0, 0] = 1
    samples[1, 0, 0, 4, 4] = 1
    target = torch.ones(1, 1, 5, 5)
    assert pool_regularization(samples, target).item() == 0
    # mean-first would have a pooled prediction of .5 and nonzero loss.
    assert torch.nn.functional.max_pool2d(samples.mean(0), 5).item() == .5


def test_constant_motion_has_zero_sobel_penalty():
    target = torch.ones(1, 2, 8, 8)
    assert motion_regularization(torch.full((1, 2, 2, 8, 8), 7.), target).item() == 0


def test_invalid_geometry_and_noise_fail_explicitly():
    model = small_model()
    with pytest.raises(ValueError, match="divisible"):
        model(torch.ones(1, 3, 65, 64))
    with pytest.raises(ValueError, match="noise"):
        model(history(), torch.ones(1, 4, 4, 4))
    with pytest.raises(ValueError, match="positive integer"):
        ModelConfig(output_steps=0)


def test_bilinear_auxiliary_required_not_silently_evolution_only_wrong_gradient():
    target = torch.zeros(1, 2, 8, 8)
    with pytest.raises(ValueError, match="auxiliary"):
        evolution_loss(target, target, torch.zeros(1, 2, 2, 8, 8))
