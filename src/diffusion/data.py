"""Read the existing case split; replace v with cached raw w before standardizing."""
import hashlib
import importlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

COUNTS = {'train': 224, 'val': 48, 'test': 50}


def inventory(data_root, height, w_root):
    folders = {}
    for split, count in COUNTS.items():
        cases = sorted((Path(data_root) / f'yplus_{height}_data' / split).glob('*'))
        cases = [p for p in cases if p.is_dir()]
        if len(cases) != count:
            raise ValueError(f'{split}: expected {count} cases, got {len(cases)}')
        for case in cases:
            if len(list(case.glob('*.npz'))) != 28:
                raise ValueError(f'Expected 28 patches in {case}')
            if not (Path(w_root) / f'yplus_{height}' / f'{case.name}.npy').is_file():
                raise FileNotFoundError(f'Missing w for {case.name}')
        folders[split] = cases
    names = [case.name for cases in folders.values() for case in cases]
    if len(names) != len(set(names)):
        raise ValueError('Case identities overlap between splits')
    return folders


def location(path):
    return tuple(map(int, Path(path).stem.split('_')[-2:]))


def raw_patch(path, w_root, height, target):
    path = Path(path)
    with np.load(path) as archive:
        velocity = archive['feature'][:2].astype(np.float32).copy()
        quantity = (archive['target'][:2] if target is None else archive['target'][target:target+1]).astype(np.float32).copy()
    h, w = location(path)
    full_w = np.load(Path(w_root) / f'yplus_{height}' / f'{path.parent.name}.npy', mmap_mode='r')
    velocity[1] = full_w[h:h+velocity.shape[-2], w:w+velocity.shape[-1]]
    return np.concatenate([velocity, quantity], axis=0)


def raw_case(case, w_root, height, target):
    result = np.zeros((4 if target is None else 3, 1400, 800), dtype=np.float32)
    count = np.zeros((1, 1400, 800), dtype=np.float32)
    for path in sorted(Path(case).glob('*.npz')):
        patch = raw_patch(path, w_root, height, target)
        h, w = location(path)
        ph, pw = patch.shape[-2:]
        result[:, h:h+ph, w:w+pw] += patch
        count[:, h:h+ph, w:w+pw] += 1
    if (count == 0).any():
        raise ValueError(f'Incomplete full-field coverage: {case}')
    result /= count
    if not np.isfinite(result).all():
        raise ValueError(f'Nonfinite full field: {case}')
    return result


def training_statistics(cases, w_root, height, target):
    total = np.zeros(4 if target is None else 3, dtype=np.float64)
    squared = total.copy()
    size = 0
    for index, case in enumerate(cases):
        field = raw_case(case, w_root, height, target).astype(np.float64)
        total += field.sum(axis=(1, 2))
        squared += np.square(field).sum(axis=(1, 2))
        size += field.shape[1] * field.shape[2]
        if (index+1) % 16 == 0:
            print(f'Normalization: {index+1}/{len(cases)} training cases', flush=True)
    mean = total / size
    std = np.sqrt(np.maximum(squared / size - mean**2, 0))
    if (std <= 0).any():
        raise ValueError('Constant normalization channel')
    return {'mean': mean.tolist(), 'std': std.tolist(), 'scope': 'training cases only; uniform full-field pixels'}


def existing_statistics(cases, w_root, height, target):
    """Use exactly the fixed statistics of scripts/retrain_uw.py, including float32 w."""
    if target is None:
        cf = existing_statistics(cases, w_root, height, 0)
        p = existing_statistics(cases, w_root, height, 1)
        np.testing.assert_array_equal(cf['mean'][:2], p['mean'][:2])
        np.testing.assert_array_equal(cf['std'][:2], p['std'][:2])
        return dict(cf, mean=cf['mean'] + p['mean'][2:], std=cf['std'] + p['std'][2:],
                    sources=list(dict.fromkeys(cf['sources'] + p['sources'])))
    source = Path(w_root) / f'yplus_{height}' / 'metadata.json'
    metadata = json.loads(source.read_text())
    if {split: [p.name for p in group] for split, group in cases.items()} != metadata['cases']:
        raise ValueError('UW metadata case identities differ from the current split')
    module_name = 'config_p_down' if target == 1 else 'config_cf_down'
    module = importlib.import_module(module_name)
    feature_mean, feature_std, target_mean, target_std = module.get_normalization(f'yplus_{height}')
    feature_mean, feature_std = feature_mean.flatten().clone(), feature_std.flatten().clone()
    feature_mean[1], feature_std[1] = metadata['w_mean'], metadata['w_std']
    mean = [feature_mean[0].item(), feature_mean[1].item(), target_mean.flatten()[0].item()]
    std = [feature_std[0].item(), feature_std[1].item(), target_std.flatten()[0].item()]
    if not np.isfinite(mean + std).all() or min(std) <= 0:
        raise ValueError('Invalid fixed normalization statistics')
    return dict(mean=mean, std=std, epsilon=1e-8,
                scope=metadata['normalization_scope'],
                sources=[str(Path(module.__file__).resolve()), str(source.resolve())],
                protocol='same fixed normalization as current UW deterministic retraining')


def standardized(field, statistics):
    return torch.from_numpy((field - np.asarray(statistics['mean'], dtype=np.float32)[:, None, None]) /
                            (np.asarray(statistics['std'], dtype=np.float32)[:, None, None] + statistics.get('epsilon', 0.)))


def make_condition(velocity, quantity, density, generator=None):
    if not 0 <= density <= 1:
        raise ValueError('Observation density must be in [0,1]')
    mask = torch.zeros_like(quantity)
    n = round(density * quantity.numel())
    if n == 0:
        # No RNG consumption and no target values in velocity-only conditions.
        return torch.cat([velocity, mask, mask.clone()], dim=0)
    indices = torch.randperm(quantity.numel(), generator=generator, device=quantity.device)[:n]
    mask.reshape(-1)[indices] = 1
    return torch.cat([velocity, mask, mask * quantity], dim=0)


def grid_condition(velocity, quantity, block_size, origin=(0, 0), full_shape=(1400, 800)):
    """Same global sensor lattice and final-row/column coverage as sample_and_interpolate."""
    if block_size < 1:
        raise ValueError('block_size must be positive')
    h, w = origin
    rows = torch.arange(h, h+quantity.shape[-2], device=quantity.device)
    cols = torch.arange(w, w+quantity.shape[-1], device=quantity.device)
    selected_rows = (rows % block_size == 0) | (rows == full_shape[0]-1)
    selected_cols = (cols % block_size == 0) | (cols == full_shape[1]-1)
    mask = (selected_rows[:, None] & selected_cols[None, :]).to(quantity.dtype)[None]
    return torch.cat([velocity, mask, mask * quantity], dim=0)


class WallPatches(Dataset):
    def __init__(self, cases, w_root, height, target, statistics, density, block_size=None):
        self.files = [p for case in cases for p in sorted(case.glob('*.npz'))]
        self.w_root, self.height, self.target = w_root, height, target
        self.statistics, self.density = statistics, density
        self.block_size = block_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        field = standardized(raw_patch(self.files[index], self.w_root, self.height, self.target), self.statistics)
        if self.target is None:
            if self.density != 0 or self.block_size is not None:
                raise ValueError('Joint UW baseline must not receive wall observations')
            return field[:2], field[2:]
        condition = (grid_condition(field[:2], field[2:], self.block_size, location(self.files[index]))
                     if self.block_size is not None else make_condition(field[:2], field[2:], self.density))
        return condition, field[2:]


def case_tensors(case, w_root, height, target, statistics, density, seed, block_size=None):
    field = standardized(raw_case(case, w_root, height, target), statistics)
    if target is None:
        if density != 0 or block_size is not None:
            raise ValueError('Joint UW baseline must not receive wall observations')
        return field[:2], field[2:]
    if block_size is not None:
        return grid_condition(field[:2], field[2:], block_size), field[2:]
    digest = hashlib.sha256(f'{seed}:{Path(case).name}'.encode()).digest()
    generator = torch.Generator().manual_seed(int.from_bytes(digest[:8], 'little') % (2**63-1))
    return make_condition(field[:2], field[2:], density, generator), field[2:]
