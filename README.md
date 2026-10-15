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
It is **not** the original Jinzun architecture or checkpoint-compatible with it.

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

## Required data (not distributed)

The loader currently expects the original cached **224 train / 48 validation /
50 test case split**, 28 patches per case, full shape (1400,800), patch size 256.
It does not generate or repartition data.

```text
data/
  cf_p/yplus_15_data/{train,val,test}/CASE/patch_H_W.npz
  uw_fields/yplus_15/CASE.npy
  uw_fields/yplus_15/metadata.json
```

Repeat for heights 30,50,100. Each patch NPZ contains raw `feature` channels
[u,v,...] and raw `target` channels [Cf,p], shaped (C,256,256). H,W in its
filename are full-field patch origins. The separate CASE.npy is the raw
full-field **w only**, shape (1400,800); it replaces cached feature channel 1
before normalization. Paths must use the same orientation as the cached data.

For `--normalization existing`, metadata.json must provide `cases`
(with sorted case-name lists for train,val,test), `w_mean`, `w_std`, and
`normalization_scope`. The lightweight config modules contain only the original
fixed u/target constants. **These historical statistics include held-out data.**
This mode is retained for matching legacy deterministic comparisons, not strict
independent-test claims. Recommended for a fresh independent-test protocol:
`--normalization train-only`, which estimates all channel statistics on training
cases only. Inference reuses checkpoint statistics. Do not mix the two protocols.

## Train

Check GPU availability first. Defaults: AdamW, learning rate 1e-4, batch 8,
200 epochs, seed 42, cosine LR decay to 1e-5, validation every 5 epochs.
Validation samples complete fields; weights are selected by validation MSE.

Pressure, fixed grid (block 30 is approximately 0.12% observations):

```bash
python train_diffusion.py --target p --yplus 15 --device cuda:0 --amp \
  --sensor-layout grid --block-size 30 --normalization train-only
```

Velocity-only single pressure target:

```bash
python train_diffusion.py --target p --yplus 15 --device cuda:0 --amp \
  --sensor-layout random --density 0 --normalization train-only
```

Joint Cf and pressure, velocity-only (change yplus for other heights):

```bash
python train_diffusion.py --target joint --yplus 15 --device cuda:0 --amp \
  --sensor-layout random --density 0 --normalization train-only
```

Training writes a timestamped directory under output/diffusion with command,
statistics, data identities, architecture, and checkpoints. No training starts
merely by importing this repository. The scripts/run_* helpers preserve the
legacy existing-normalization protocol for reproduction; use the direct entry
point above for training-only normalization.

## Predict

Single target on cached test cases:

```bash
python predict_diffusion.py --checkpoint output/diffusion/RUN/best.pt \
  --split test --device cuda:0 --samples 4 --output output/predictions/pressure_RUN
```

Joint target:

```bash
python scripts/predict_joint_diffusion.py --checkpoint output/diffusion/RUN/best.pt \
  --device cuda:0 --samples 4 --output output/predictions/joint_RUN
```

Single-target external inference requires an NPZ with same-shaped 2D arrays
u,w,mask,observations in original dataset units. Only sensor values are used;
no complete true wall field is required. Keep the original dataset's physical
units and coordinate convention.

```bash
python predict_diffusion.py --checkpoint output/diffusion/RUN/best.pt \
  --condition measurements.npz --device cuda:0 --samples 4 \
  --output output/predictions/external_RUN
```

Predictions use full-field network inference, not patch-wise inference.
Multiple samples yield a mean field and pointwise sampling standard deviation;
one sample cannot estimate uncertainty. Joint evaluation records first-sample
and ensemble-mean metrics separately. A new output directory is required for
each scientifically distinct experiment.

## Repository layout

- src/diffusion/: network, diffusion process, data and observation construction.
- train_diffusion.py / predict_diffusion.py: single/joint training and single inference.
- scripts/predict_joint_diffusion.py: joint inference.
- scripts/run_joint_diffusion.py: independent height jobs across available GPUs.
- scripts/run_diffusion_trial.py: single-target training/evaluation wrapper.
- tests/test_diffusion.py: CPU tests for algebra, masks, channels and gradients.

The data adapter remains specific to the cached TCF data layout. This release
does not include DNS export/preparation or TBL transfer pipelines. There is no
new license grant included; choose an appropriate license with all code owners
before representing the repository as licensed open-source software.
