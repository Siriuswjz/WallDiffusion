"""Train single-target or joint direct conditional diffusion on the cached TCF split."""
import argparse
import datetime
import json
from pathlib import Path
import random
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.diffusion import ConditionalUNet, GaussianDiffusion
from src.diffusion.data import inventory, training_statistics, existing_statistics, WallPatches, case_tensors

ROOT = Path(__file__).resolve().parent


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target', choices=['p', 'cf', 'joint'], default='p')
    p.add_argument('--yplus', type=int, choices=[15, 30, 50, 100], default=15)
    p.add_argument('--data-root', type=Path, default=ROOT/'data/cf_p')
    p.add_argument('--w-root', type=Path, default=ROOT/'data/uw_fields')
    p.add_argument('--output-root', type=Path, default=ROOT/'output/diffusion')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--density', type=float, default=.0102)
    p.add_argument('--sensor-layout', choices=['grid', 'random'], default='grid')
    p.add_argument('--block-size', type=int, default=10,
                   help='Fixed global lattice spacing; final row/column included; density ignored in grid mode')
    p.add_argument('--normalization', choices=['existing', 'train-only'], default='existing',
                   help='existing: same fixed statistics as current UW models; train-only: recompute from training cases')
    p.add_argument('--base-channels', type=int, default=32)
    p.add_argument('--attention-depth', type=int, default=2)
    p.add_argument('--heads', type=int, default=8)
    p.add_argument('--timesteps', type=int, default=1000)
    p.add_argument('--prediction-type', choices=['epsilon', 'v'], default='v',
                   help='v: stable auxiliary diffusion target, unrelated to the physical velocity component v')
    p.add_argument('--sampling-steps', type=int, default=50)
    p.add_argument('--eval-interval', type=int, default=5)
    p.add_argument('--enforce-observations', action='store_true')
    p.add_argument('--amp', action='store_true')
    return p


def main():
    args = parser().parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.eval_interval < 1 or args.block_size < 1 or not 0 <= args.density <= 1:
        raise ValueError('Invalid training settings')
    if args.target == 'joint' and (args.density != 0 or args.sensor_layout != 'random' or args.enforce_observations):
        raise ValueError('Joint UW baseline requires --density 0 --sensor-layout random and no hard observations')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(args.seed)
    cases = inventory(args.data_root, args.yplus, args.w_root)
    output = args.output_root / f'{datetime.datetime.now():%Y%m%d_%H%M%S_%f}_{args.target}_y{args.yplus}_uw'
    output.mkdir(parents=True, exist_ok=False)
    target = None if args.target == 'joint' else (1 if args.target == 'p' else 0)
    statistics = (existing_statistics(cases, args.w_root, args.yplus, target)
                  if args.normalization == 'existing'
                  else training_statistics(cases['train'], args.w_root, args.yplus, target))
    architecture = dict(base_channels=args.base_channels, attention_depth=args.attention_depth, heads=args.heads)
    if target is None:
        architecture.update(target_channels=2, condition_channels=2)
    block_size = args.block_size if args.sensor_layout == 'grid' else None
    observation_protocol = ('fixed global grid matching sample_and_interpolate; final row/column included; noise-free'
                            if block_size is not None else
                            'uniform random without replacement; noise-free; new patch mask per training access; fixed full-field validation mask')
    if target is None:
        observation_protocol = 'none; velocity-only UW condition; no target information in condition'
    protocol = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    command=[sys.executable, *sys.argv], components=['u', 'w'], target=args.target, statistics=statistics,
                    cases={k: [p.name for p in v] for k, v in cases.items()}, architecture=architecture,
                    observations=observation_protocol,
                    selection='lowest full-field standardized validation MSE; all 48 validation cases',
                    objective=f'{args.prediction_type}-prediction MSE', normalization_scope=statistics['scope'])
    if target is None:
        protocol.update(target_channels=['cf','p'],condition_channels=['u','w'],
                        loss_reduction='mean over batch, both standardized target channels and pixels; equal channel weight')
    (output/'protocol.json').write_text(json.dumps(protocol, indent=2))
    print(f'Run: {output}', flush=True)
    model = ConditionalUNet(**architecture).to(device)
    diffusion = GaussianDiffusion(args.timesteps, args.prediction_type).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.lr/10)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp and device.type == 'cuda')
    loader = DataLoader(WallPatches(cases['train'], args.w_root, args.yplus, target, statistics, args.density, block_size),
                        batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                        worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(args.seed))
    best = float('inf')
    with (output/'metrics.jsonl').open('a', buffering=1) as log:
        for epoch in range(1, args.epochs+1):
            model.train()
            total, count = 0., 0
            for batch_index, (condition, clean) in enumerate(loader, 1):
                condition, clean = condition.to(device), clean.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                    loss = diffusion.loss(model, clean, condition)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite training loss')
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                scaler.step(optimizer); scaler.update()
                total += loss.item() * len(clean); count += len(clean)
                if batch_index % 100 == 0:
                    print(f'Epoch {epoch}/{args.epochs}, batch {batch_index}/{len(loader)}, {args.prediction_type}-prediction MSE={total/count:.6f}', flush=True)
            row = dict(epoch=epoch, prediction_mse=total/count, prediction_type=args.prediction_type,
                       lr=optimizer.param_groups[0]['lr'])
            scheduler.step()
            if epoch % args.eval_interval == 0 or epoch == args.epochs:
                model.eval()
                errors = []
                for case in cases['val']:
                    condition, clean = case_tensors(case, args.w_root, args.yplus, target, statistics, args.density, args.seed, block_size)
                    predicted = diffusion.sample(model, condition[None].to(device), args.sampling_steps,
                                                 args.seed, args.enforce_observations).cpu()[0]
                    errors.append((predicted-clean).square().mean().item())
                row['val_full_field_standardized_mse'] = float(np.mean(errors))
                if not np.isfinite(row['val_full_field_standardized_mse']):
                    raise RuntimeError('Nonfinite validation result')
                checkpoint = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                  scheduler=scheduler.state_dict(), epoch=epoch, protocol=protocol,
                                  val_full_field_standardized_mse=row['val_full_field_standardized_mse'])
                if row['val_full_field_standardized_mse'] < best:
                    best = row['val_full_field_standardized_mse']
                    torch.save(checkpoint, output/'best.pt')
                torch.save(checkpoint, output/'last.pt')
            line = json.dumps(row)
            print(line, flush=True); log.write(line+'\n')
    print(f'Completed training; best validation MSE={best:.8g}; checkpoint={output / "best.pt"}', flush=True)


if __name__ == '__main__':
    main()
