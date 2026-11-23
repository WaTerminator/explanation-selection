"""Search for admissible continuations of the capture-timing runs that still change a label.

The source is the output of run_capture_timing.py at the step size of the selection experiment. From saved
states between the last label change of a run and its capture, every continuation applies Grams with
0.25 I <= K_t <= 4 I that depend on the current predictions and the counts only. A continuation that changes a
label shows that the labels were still reversible at its start state. A search that finds none shows nothing.

Three searches are run in turn: a greedy rule, Gram sequences optimized by gradient descent, and a refinement
of the optimized sequences toward later start states. evaluate_capture_timing.py replays every successful
continuation independently.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from run_cifar_identifiable_traps import json_write
from run_cifar_llp import candidates, quantities
from run_evidence_capture import adverse_direction


@torch.jit.script
def state(y: Tensor, codes: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
    """Residual of the soft-label update and global log evidence."""
    grouped = y.reshape(-1, codes.shape[0], 4)
    scores = torch.einsum('nbi,bci->nbc', [grouped, codes])
    scores = scores.masked_fill(~valid.unsqueeze(0), float('-inf'))
    posterior = torch.softmax(scores, dim=-1)
    target = torch.einsum('nbc,bci->nbi', [posterior, codes]).reshape_as(y)
    residual = 2 * (target - torch.sigmoid(y))
    log_evidence = (torch.logsumexp(scores, dim=-1)
                    - torch.nn.functional.softplus(grouped).sum(-1)).sum(-1)
    return residual, log_evidence


@torch.jit.script
def direction(residual: Tensor, target: Tensor, sign: Tensor) -> Tensor:
    """Slow direction u of K = 4I - 3.75 uu' that minimizes the drift of the targeted signed logit."""
    norm = torch.linalg.vector_norm(residual, dim=1, keepdim=True).clamp_min(1e-300)
    unit = torch.zeros_like(residual).scatter(1, target[:, None], sign[:, None])
    raw = residual / norm + unit
    length = torch.linalg.vector_norm(raw, dim=1, keepdim=True)
    fallback = torch.zeros_like(residual).scatter(
        1, ((target + 1) % residual.shape[1])[:, None], 1.)
    return torch.where(length < 1e-12, fallback, raw / length.clamp_min(1e-300))


def rollout(initial: Tensor, controls: Tensor, target: Tensor, signs: Tensor,
            codes: Tensor, valid: Tensor, horizon: int, block: int, eta: float,
            center: float, radius: float) -> tuple[Tensor, Tensor]:
    """Final signed logit of the targeted label and the smallest signed logit reached along the way."""
    y = initial
    anchor = torch.sign(initial)
    minimum = torch.full((len(y),), float('inf'), device=y.device, dtype=y.dtype)
    normalized = torch.nn.functional.normalize(controls, dim=-1)
    for step in range(horizon):
        residual, _ = state(y, codes, valid)
        change = center*residual + radius*torch.linalg.vector_norm(residual, dim=1, keepdim=True)*normalized[step//block]
        y = y + eta*change
        minimum = torch.minimum(minimum, (anchor*y.detach()).min(dim=1).values)
    return signs*y.gather(1, target[:, None])[:, 0], minimum


def implementation_check(device: str) -> None:
    """Check the posterior, the optimal Gram action, the Gram spectrum and the gradient of the controls."""
    rng = np.random.default_rng(2026110901)
    counts = np.array([1, 2, 3, 1, 2, 3])
    codes, valid = candidates(counts)
    y = rng.normal(size=(13, 24))
    reference = quantities(y, codes, valid)
    ty = torch.tensor(y, dtype=torch.float64, device=device)
    residual, log_evidence = state(ty, torch.tensor(codes, device=device),
                                   torch.tensor(valid, device=device))
    np.testing.assert_allclose(residual.cpu(), reference.residual, atol=1e-13)
    np.testing.assert_allclose(log_evidence.cpu(), reference.log_bag_evidence.sum(1), atol=1e-13)
    target = rng.integers(0, 24, size=13)
    sign = np.where(y[np.arange(13), target] > 0, 1., -1.)
    slow = direction(residual, torch.tensor(target, device=device),
                     torch.tensor(sign, device=device)).cpu().numpy()
    metric = 4*np.eye(24)[None] - 3.75*slow[:, :, None]*slow[:, None, :]
    values = np.linalg.eigvalsh(metric)
    np.testing.assert_allclose(values[:, 0], .25, atol=1e-13)
    np.testing.assert_allclose(values[:, 1:], 4, atol=1e-13)
    action = np.einsum('bij,bj->bi', metric, reference.residual)
    optimum = 2.125*sign*reference.residual[np.arange(13), target] - 1.875*np.linalg.norm(reference.residual, axis=1)
    np.testing.assert_allclose(sign*action[np.arange(13), target], optimum, atol=1e-13)
    weakest = np.abs(y).argmin(1)
    equivalent = direction(residual, torch.tensor(weakest, device=device),
                           torch.tensor(np.sign(y[np.arange(13), weakest]), device=device))
    np.testing.assert_allclose(equivalent.cpu(), adverse_direction(y, reference.residual, np.sign(y)), atol=1e-13)

    rng = np.random.default_rng(2026110902)
    codes, valid = candidates(np.array([1]))
    initial = torch.tensor(rng.normal(size=(2, 4)), dtype=torch.float64, device=device)
    controls = torch.tensor(rng.normal(size=(2, 2, 4)), dtype=torch.float64, device=device, requires_grad=True)
    target_t = torch.tensor([0, 1], device=device)
    signs = torch.sign(initial.gather(1, target_t[:, None])[:, 0])
    tcodes, tvalid = torch.tensor(codes, device=device), torch.tensor(valid, device=device)
    args = (target_t, signs, tcodes, tvalid, 3, 2, .032, 2.125, 1.875)
    loss, _ = rollout(initial, controls, *args)
    gradient = torch.autograd.grad(loss.sum(), controls)[0]
    for index in ((0, 0, 1), (1, 1, 3)):
        plus, minus = controls.detach().clone(), controls.detach().clone()
        plus[index] += 1e-5
        minus[index] -= 1e-5
        with torch.no_grad():
            numeric = (rollout(initial, plus, *args)[0].sum()-rollout(initial, minus, *args)[0].sum())/2e-5
        np.testing.assert_allclose(float(gradient[index]), float(numeric), rtol=1e-5, atol=1e-9)
    residual, _ = state(initial, tcodes, tvalid)
    r, v = residual.detach().cpu().numpy()[0], controls.detach().cpu().numpy()[0, 0]
    v /= np.linalg.norm(v)
    u = r/np.linalg.norm(r)-v
    u /= np.linalg.norm(u)
    gram = 4*np.eye(4)-3.75*np.outer(u, u)
    np.testing.assert_allclose(gram@r, 2.125*r+1.875*np.linalg.norm(r)*v, atol=1e-13)
    print(json.dumps({'implementation_check': 'passed', 'device': device}), flush=True)


@torch.no_grad()
def greedy(source: Path, settings: dict, common: dict, output: Path, device: str) -> None:
    output.mkdir(parents=True, exist_ok=False)
    with np.load(source/'summary.npz', allow_pickle=False) as data:
        capture, last, counts = (data[k] for k in ('capture', 'last_change', 'counts'))
        eta = float(data['eta'])
    history = np.load(source/'baseline.npy', mmap_mode='r', allow_pickle=False)
    width = history.shape[-1]
    assert np.all(capture > last)
    assert eta*common['kappa_max']*(width+1) < 4
    start_pairs = [(i, t) for i in range(len(capture)) for t in sorted(set(
        [int(last[i] + f*(capture[i]-last[i])) for f in settings['fractions_after_last_change']]
        + ([int(capture[i]-1)] if settings['include_last_state_before_capture'] else [])))]
    source_index = np.repeat([p[0] for p in start_pairs], width)
    source_time = np.repeat([p[1] for p in start_pairs], width)
    targets = np.tile(np.arange(width), len(start_pairs))
    initial = np.asarray(history[source_time, source_index]).copy()
    assert np.isfinite(initial).all() and np.all(initial != 0)
    codes, valid = candidates(counts)
    tcodes = torch.tensor(codes, dtype=torch.float64, device=device)
    tvalid = torch.tensor(valid, device=device)
    current = torch.tensor(initial, dtype=torch.float64, device=device)
    anchor = torch.sign(current)
    target = torch.tensor(targets, device=device)
    signs = anchor.gather(1, target[:, None])[:, 0]
    residual, evidence = state(current, tcodes, tvalid)
    assert bool(torch.all(evidence <= -np.log(2)+1e-10))
    first_flip = torch.full((len(current),), -1, device=device, dtype=torch.int64)
    first_capture = first_flip.clone()
    active = torch.arange(len(current), device=device)
    minimum_increment = 0.
    started = time.monotonic()
    for step in range(1, settings['horizon']+1):
        if not len(active):
            break
        old = current[active]
        old_residual, old_evidence = state(old, tcodes, tvalid)
        slow = direction(old_residual, target[active], signs[active])
        change = (common['kappa_max']*old_residual
                  -(common['kappa_max']-common['kappa_min'])*slow*(slow*old_residual).sum(1, keepdim=True))
        new = old + eta*change
        _, new_evidence = state(new, tcodes, tvalid)
        increment = float((new_evidence-old_evidence).min())
        minimum_increment = min(minimum_increment, increment)
        if increment < -1e-10 or not bool(torch.isfinite(new).all()):
            raise ArithmeticError(f'Invalid admissible update at step {step}: increment={increment}')
        current[active] = new
        flip = (anchor[active]*new).min(1).values < -common['label_change_margin']
        crossed = new_evidence > -np.log(2)+1e-11
        first_flip[active[flip]] = step
        first_capture[active[crossed]] = step
        active = active[~(flip | crossed)]
        if step % 100 == 0 or not len(active):
            print(json.dumps({'search': 'greedy', 'update': step, 'active': len(active),
                              'label_changing_continuations': int((first_flip > 0).sum()),
                              'seconds': time.monotonic()-started}), flush=True)
    flip_np = first_flip.cpu().numpy()
    np.savez_compressed(output/'continuations.npz', source_index=source_index, source_time=source_time,
                        targets=targets, initial=initial, first_flip=flip_np,
                        first_capture=first_capture.cpu().numpy())
    json_write(output/'completed.json', {'start_states': len(start_pairs), 'continuations': len(current),
               'label_changing_continuations': int((flip_np > 0).sum()),
               'runs_with_a_label_changing_continuation': int(len(np.unique(source_index[flip_np > 0]))),
               'minimum_log_evidence_increment': minimum_increment, 'seconds': time.monotonic()-started})


def optimize(source: Path, settings: dict, common: dict, output: Path, device: str, warm_start: Path | None) -> None:
    output.mkdir(parents=True, exist_ok=False)
    with np.load(source/'summary.npz', allow_pickle=False) as data:
        capture, last, counts = (data[k] for k in ('capture', 'last_change', 'counts'))
        eta = float(data['eta'])
    history = np.load(source/'baseline.npy', mmap_mode='r', allow_pickle=False)
    width = history.shape[-1]
    assert eta*common['kappa_max']*(width+1) < 4
    codes, valid = candidates(counts)
    tcodes = torch.tensor(codes, device=device, dtype=torch.float64)
    tvalid = torch.tensor(valid, device=device)
    indices = np.repeat(np.arange(len(capture)), width)
    targets = np.tile(np.arange(width), len(capture))
    target = torch.tensor(targets, device=device)
    center = (common['kappa_min']+common['kappa_max'])/2
    radius = (common['kappa_max']-common['kappa_min'])/2
    block = settings['block_updates']
    started = time.monotonic()
    summaries: list[dict[str, int | float | str]] = []
    warm: Tensor | None = None
    if warm_start is not None:
        with np.load(warm_start, allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['source_index'], indices)
            np.testing.assert_array_equal(saved['targets'], targets)
            warm = torch.tensor(saved['controls'], device=device, dtype=torch.float64)
    for fraction in settings['fractions_after_last_change']:
        times = np.repeat((last+fraction*(capture-last)).astype(np.int64), width)
        initial = torch.tensor(np.asarray(history[times, indices]), device=device, dtype=torch.float64)
        signs = torch.sign(initial.gather(1, target[:, None])[:, 0])
        for horizon in settings['horizons']:
            blocks = (horizon+block-1)//block
            base = torch.zeros_like(initial).scatter(1, target[:, None], -signs[:, None])
            if warm is not None:
                assert warm.shape == (blocks, len(initial), width)
            controls = (warm if warm is not None else base.unsqueeze(0).repeat(blocks, 1, 1)).detach().clone().requires_grad_(True)
            optimizer = torch.optim.Adam([controls], lr=settings['control_step_size'])
            best_margin = torch.full((len(initial),), float('inf'), device=device, dtype=initial.dtype)
            best_controls = controls.detach().clone()
            for iteration in range(settings['iterations']+1):
                optimizer.zero_grad(set_to_none=True)
                value, margin = rollout(initial, controls, target, signs, tcodes, tvalid,
                                        horizon, block, eta, center, radius)
                if not bool(torch.isfinite(value).all()):
                    raise FloatingPointError('Nonfinite controlled trajectory')
                improved = margin < best_margin
                with torch.no_grad():
                    best_margin[improved] = margin[improved]
                    best_controls[:, improved] = controls[:, improved]
                if iteration % 20 == 0 or iteration == settings['iterations']:
                    success = best_margin < -common['label_change_margin']
                    print(json.dumps({'search': output.name, 'fraction': fraction, 'horizon': horizon, 'iteration': iteration,
                                      'label_changing_continuations': int(success.sum()),
                                      'runs_with_a_label_changing_continuation': int(success.reshape(-1, width).any(1).sum()),
                                      'seconds': time.monotonic()-started}), flush=True)
                if iteration < settings['iterations']:
                    value.sum().backward()
                    optimizer.step()
            tag = f'fraction_{fraction}_horizon_{horizon}'
            np.savez_compressed(output/f'{tag}.npz', source_index=indices, source_time=times,
                                targets=targets, controls=best_controls.cpu().numpy(),
                                minimum_signed_margin=best_margin.cpu().numpy())
            if warm_start is not None:
                warm = best_controls.detach().clone()
            success = (best_margin < -common['label_change_margin']).cpu().numpy()
            summaries.append({'tag': tag, 'fraction': fraction, 'horizon': horizon, 'block_updates': block,
                              'label_changing_continuations': int(success.sum()),
                              'runs_with_a_label_changing_continuation': int(success.reshape(-1, width).any(1).sum())})
    json_write(output/'completed.json', {'seconds': time.monotonic()-started, 'searches': summaries})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True,
                        help='output of run_capture_timing.py at the fraction named by the protocol')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    common = json.loads(args.protocol.read_text(encoding='utf-8'))['reversibility']
    record = json.loads((args.source/'completed.json').read_text(encoding='utf-8'))
    if record['fraction_of_the_step_bound'] != common['fraction_of_the_step_bound']:
        raise ValueError('The source was run at a different step size')
    torch.set_num_threads(1)
    implementation_check(args.device)
    args.output.mkdir(parents=True, exist_ok=False)
    greedy(args.source, common['greedy'], common, args.output/'greedy', args.device)
    optimize(args.source, common['optimized'], common, args.output/'optimized', args.device, None)
    warm = common['refined']['warm_start_from_optimized']
    optimize(args.source, common['refined'], common, args.output/'refined', args.device,
             args.output/'optimized'/f"fraction_{warm['fraction']}_horizon_{warm['horizon']}.npz")


if __name__ == '__main__':
    main()
