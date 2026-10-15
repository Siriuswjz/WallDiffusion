# WallDiffusion

Conditional diffusion for reconstructing turbulent wall skin-friction coefficient
(Cf) and pressure (p) from wall-parallel velocity components **u,w**, optionally
conditioned on sparse wall measurements.

This is a standalone extraction of the diffusion implementation from Final_CF_P.
No DNS data, trained checkpoints, predictions, private configuration, or parent
repository history is included. Historical numerical results are not bundled;
training artifacts must be regenerated.

## Model and scope

A timestep-conditioned U-Net with bottleneck attention learns a DDPM cosine-noise
process. Defaults use auxiliary **v-prediction** (unrelated to physical wall-normal
velocity v), 1,000 diffusion timesteps and 50 deterministic DDIM sampling steps.

- Single-target model: noisy wall quantity, timestep, and conditions
  [u,w,mask,masked observations]. No spatial interpolation of observations.
- Joint model: two noisy target channels [Cf,p], timestep and conditions [u,w].
  Joint sparse conditioning is not implemented.
- Grid sensors include the final full-field row and column. Optional random
  masks change during training and are case-seeded during evaluation.
- Hard observation projection is optional, single-target only, and heuristic;
  it is not an exact posterior sampler.
- Ensemble spread measures sampling variability, not validated uncertainty
  calibration. Report single-sample and ensemble-mean errors separately.

## Install and test

Use Python 3.10+ with an appropriate PyTorch build for your CUDA environment:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

## Data

Provide the cached TCF split (224 training, 48 validation and 50 test cases)
under `data/cf_p/`, with raw w fields under `data/uw_fields/`. Data and
checkpoints are not distributed.

Use `--normalization train-only` for training-set statistics. The default
`existing` mode reproduces legacy fixed statistics that include held-out data.

## Quick start

Train pressure reconstruction with sparse observations:

```bash
python train_diffusion.py --target p --yplus 15 --amp --block-size 30 --normalization train-only
```

For velocity-only conditioning, add `--sensor-layout random --density 0`.
Use `--target cf` for skin friction or `--target joint` with velocity-only
conditioning to predict both quantities. Supported heights: 15, 30, 50 and 100.

Predict from a single-target checkpoint:

```bash
python predict_diffusion.py --checkpoint output/diffusion/RUN/best.pt --samples 4 --output output/predictions/RUN
```

For joint predictions use `scripts/predict_joint_diffusion.py`. Outputs include
sample means and pointwise sampling standard deviations. Run any entry point
with `--help` for additional options.

## Code

- `src/diffusion/`: model, diffusion process and data loader.
- `train_diffusion.py`, `predict_diffusion.py`: training and prediction.
- `scripts/`: multi-GPU experiments and joint prediction.
- `tests/`: CPU checks.
