"""Full-field conditional DDIM inference from external observations or cached cases."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
from src.diffusion import ConditionalUNet, GaussianDiffusion
from src.diffusion.data import inventory, case_tensors


def external_condition(path, statistics):
    """NPZ values u,w,observations are RAW; mask is binary; no truth needed."""
    with np.load(path) as archive:
        arrays = [archive[key].astype(np.float32) for key in ('u', 'w', 'mask', 'observations')]
    if any(a.ndim != 2 or a.shape != arrays[0].shape for a in arrays):
        raise ValueError('External u,w,mask,observations must have the same 2D shape')
    u, w, mask, observations = arrays
    if not np.isfinite(u).all() or not np.isfinite(w).all() or not np.isin(mask, [0, 1]).all():
        raise ValueError('Invalid velocities or nonbinary mask')
    if not np.isfinite(observations[mask == 1]).all():
        raise ValueError('Nonfinite observed values')
    mean, std = statistics['mean'], statistics['std']
    epsilon = statistics.get('epsilon', 0.)
    observed = np.zeros_like(observations)
    observed[mask == 1] = (observations[mask == 1] - mean[2]) / (std[2]+epsilon)
    return torch.from_numpy(np.stack([(u-mean[0])/(std[0]+epsilon), (w-mean[1])/(std[1]+epsilon), mask, observed]))


def metrics(predicted, truth, mask, observations):
    centered_truth = truth-truth.mean()
    centered_prediction = predicted-predicted.mean()
    variance = np.mean(centered_truth**2)
    denominator = np.sqrt(np.sum(centered_truth**2) * np.sum(centered_prediction**2))
    return dict(mse=float(np.mean((predicted-truth)**2)),
                relative_mse=float(np.mean((predicted-truth)**2)/variance) if variance else None,
                correlation=float(np.sum(centered_truth*centered_prediction)/denominator) if denominator else None,
                amplitude_ratio=float(predicted.std()/truth.std()) if truth.std() else None,
                observation_rmse=float(np.sqrt(np.mean((predicted[mask]-observations[mask])**2))) if mask.any() else None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--condition', type=Path, help='External RAW u,w,mask,observations NPZ')
    source.add_argument('--split', choices=['val', 'test'])
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--samples', type=int, default=1)
    p.add_argument('--steps', type=int, default=None)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--enforce-observations', action='store_true')
    p.add_argument('--limit', type=int, help='Diagnostic subset; omit for all cases')
    args = p.parse_args()
    if args.samples < 1 or (args.limit is not None and args.limit < 1):
        raise ValueError('samples and limit must be positive')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    protocol = checkpoint['protocol']
    settings, statistics = protocol['arguments'], protocol['statistics']
    if settings['target'] == 'joint':
        raise ValueError('Use scripts/predict_joint_diffusion.py for two-target checkpoints')
    device = torch.device(args.device)
    model = ConditionalUNet(**protocol['architecture']).to(device).eval()
    model.load_state_dict(checkpoint['model'])
    diffusion = GaussianDiffusion(settings['timesteps'], settings.get('prediction_type', 'epsilon')).to(device)
    steps = args.steps if args.steps is not None else settings['sampling_steps']
    if args.output.exists():
        raise FileExistsError(f'Choose a new result directory: {args.output}')
    args.output.mkdir(parents=True)
    if args.condition:
        names = [args.condition.stem]
        def load(index):
            return external_condition(args.condition, statistics), None
    else:
        cases = inventory(settings['data_root'], settings['yplus'], settings['w_root'])
        if {k: [c.name for c in v] for k, v in cases.items()} != protocol['cases']:
            raise ValueError('Dataset case identities differ from checkpoint')
        selected = cases[args.split][:args.limit]
        names = [c.name for c in selected]
        def load(index):
            return case_tensors(selected[index], settings['w_root'], settings['yplus'],
                                1 if settings['target'] == 'p' else 0, statistics, settings['density'], args.seed,
                                settings['block_size'] if settings.get('sensor_layout') == 'grid' else None)
    manifest = dict(command=[sys.executable, *sys.argv], device=args.device,
                    checkpoint=str(args.checkpoint.resolve()), checkpoint_epoch=checkpoint['epoch'],
                    split=args.split, external_condition=str(args.condition) if args.condition else None,
                    seed=args.seed, samples=args.samples, steps=steps, enforce_observations=args.enforce_observations,
                    target=settings['target'], components=['u', 'w'], cases=names, statistics=statistics,
                    sensor_layout=settings.get('sensor_layout', 'random'), block_size=settings.get('block_size'),
                    observed_values_source='provided external measurements' if args.condition else 'synthetic noise-free observations sampled from evaluation DNS',
                    summary='Per-case metrics on ensemble mean; uncertainty is population std across generated samples')
    (args.output/'protocol.json').write_text(json.dumps(manifest, indent=2))
    rows = []
    for index, name in enumerate(names):
        condition, clean = load(index)
        condition_gpu = condition[None].to(device)
        mean, m2 = None, None
        for sample_index in range(args.samples):
            generated = diffusion.sample(model, condition_gpu, steps, args.seed+sample_index,
                                         args.enforce_observations).cpu().numpy()[0, 0]
            generated = generated * statistics['std'][2] + statistics['mean'][2]
            if not np.isfinite(generated).all():
                raise RuntimeError(f'Nonfinite sample for {name}')
            if mean is None:
                mean = generated.astype(np.float64); m2 = np.zeros_like(mean)
            else:
                delta = generated-mean
                mean += delta/(sample_index+1)
                m2 += delta*(generated-mean)
        mask = condition[2].numpy().astype(bool)
        observations = np.zeros_like(mean)
        observations[mask] = condition[3].numpy()[mask] * statistics['std'][2] + statistics['mean'][2]
        result = dict(prediction=mean.astype(np.float32), prediction_std=np.sqrt(m2/args.samples).astype(np.float32),
                      mask=mask, observations=observations.astype(np.float32))
        if clean is not None:
            truth = clean.numpy()[0] * statistics['std'][2] + statistics['mean'][2]
            result['truth'] = truth
            row = dict(case=name, **metrics(mean, truth, mask, observations))
            rows.append(row); print(json.dumps(row), flush=True)
        np.savez_compressed(args.output/f'{name}.npz', **result)
    if rows:
        summary = {key: dict(mean=float(np.mean([r[key] for r in rows if r[key] is not None])),
                            std=float(np.std([r[key] for r in rows if r[key] is not None])))
                   for key in rows[0] if key != 'case' and any(r[key] is not None for r in rows)}
        (args.output/'metrics.json').write_text(json.dumps(dict(cases=rows, summary=summary, ddof=0), indent=2))
    print(f'Completed {len(names)} full-field cases: {args.output}', flush=True)


if __name__ == '__main__':
    main()
