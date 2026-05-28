# SA-mfa_conformer (Semantic-Aware MFA-Conformer)

This repository is built on the MFA-Conformer baseline and extends it with a semantic-aware branch and fusion mechanisms, plus utilities for visualization and multi-benchmark evaluation during training (VoxCeleb1-O/E/H, SITW).

Baseline reference:
- "MFA-Conformer: Multi-scale Feature Aggregation Conformer for Automatic Speaker Verification" (submitted to Interspeech 2022)

## Environment

Use the provided `environment.yml` (recommended), or ensure you have:
- PyTorch + PyTorch Lightning
- numpy, scipy, scikit-learn
- soundfile (for reading `.flac` in SITW)
- librosa (frontend)

## Data Preparation

### 1) VoxCeleb2 training `train.csv`

If you train with VoxCeleb2 only (recommended when evaluating VoxCeleb1-E/H), generate `data/train.csv` from the Vox2 `wav/` tree (e.g., `.../vox2/wav/id00012/.../*.wav`).

Linux example:

```bash
cd /hy-tmp/ldy/SA-mfa_conformer
mkdir -p data

python scripts/build_datalist.py \
  --extension wav \
  --dataset_dir /hy-tmp/ldy/vox/vox2/wav \
  --data_list_path data/train.csv \
  --speaker_level 1
```

- `--speaker_level 1` means the speaker label is the `idXXXX` directory (fits Vox2 layout).
- The speaker count printed by this script should match `--num_classes` used for training.

### 2) VoxCeleb1 evaluation trials (Vox1-O / Vox1-E / Vox1-H)

We convert official trial lists into the project-ready 3-column format:

`label enroll_abs_path test_abs_path`

Set `VOX1_ROOT` to your local VoxCeleb1 root directory. It must contain `wav/idXXXX/.../*.wav`:

```bash
VOX1_ROOT=/hy-tmp/ldy/vox/vox1
```

Download official lists and convert:

```bash
cd /hy-tmp/ldy/SA-mfa_conformer
mkdir -p data

wget -O data/veri_test2.txt      https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt
wget -O data/list_test_all2.txt  https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/list_test_all2.txt
wget -O data/list_test_hard2.txt https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/list_test_hard2.txt

python scripts/format_trials.py \
  --voxceleb1_root "$VOX1_ROOT" \
  --src_trials_path data/veri_test2.txt \
  --dst_trials_path data/vox1_test.txt

python scripts/format_trials.py \
  --voxceleb1_root "$VOX1_ROOT" \
  --src_trials_path data/list_test_all2.txt \
  --dst_trials_path data/vox1_E_trials.txt

python scripts/format_trials.py \
  --voxceleb1_root "$VOX1_ROOT" \
  --src_trials_path data/list_test_hard2.txt \
  --dst_trials_path data/vox1_H_trials.txt
```

Quick sanity check (ensure the generated paths exist):

```bash
awk 'NR==1{print $2; print $3}' data/vox1_E_trials.txt | xargs -I{} ls -l {}
```

### 3) SITW Dev/Eval trials (SITW is FLAC)

SITW can be requested by emailing sitw_poc@speech.sri.com. This repo supports FLAC via `soundfile`.

Generate project-ready 3-column trials (core-core condition):

```bash
python scripts/make_sitw_trials.py --repo .
```

Outputs:
- `data/sitw_dev_core-core_trials.txt`
- `data/sitw_eval_core-core_trials.txt`

## Training

### Option A: Start training with `start.sh`

Edit `start.sh` to set hyperparameters and evaluation trial lists, then run:

```bash
nohup bash start.sh > output.log 2>&1 &
```

Key settings:
- `train_csv_path`: training CSV (Vox2-only recommended)
- `num_classes`: must match the number of speakers in `train.csv`
- `trial_paths=(...)`: trials evaluated during validation (can include multiple benchmarks)
- `--eval_warmup_epochs`: for the first N epochs, only evaluate the 1st trial in `trial_paths` (default 5). Starting from epoch N, evaluate all trials.

Example (validate on Vox1-O + SITW.Dev + SITW.Eval):

```bash
trial_paths=(data/vox1_test.txt data/sitw_dev_core-core_trials.txt data/sitw_eval_core-core_trials.txt)
```

### Option B: Run `main.py` directly

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --batch_size 180 \
  --num_workers 20 \
  --max_epochs 60 \
  --embedding_dim 192 \
  --save_dir experiment_xxx \
  --encoder_name conformer_cat \
  --train_csv_path data/train.csv \
  --learning_rate 0.001 \
  --num_classes 7205 \
  --trial_paths data/vox1_test.txt data/vox1_E_trials.txt data/vox1_H_trials.txt \
  --eval_warmup_epochs 5 \
  --loss_name amsoftmax \
  --num_blocks 6 \
  --step_size 4 \
  --gamma 0.5 \
  --weight_decay 0.0000001 \
  --input_layer conv2d2 \
  --pos_enc_layer_type rel_pos
```

## Evaluation (pretrained checkpoints)

Evaluate one checkpoint on a specific trial list:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --eval \
  --save_dir outputs/eval_vox1E \
  --checkpoint_path /path/to/epoch=xx_....ckpt \
  --trial_path data/vox1_E_trials.txt \
  --encoder_name conformer_cat --num_blocks 6 --embedding_dim 192 \
  --input_layer conv2d2 --pos_enc_layer_type rel_pos \
  --num_classes 7205
```

Notes:
- `--embedding_dim / --num_blocks / --input_layer / --pos_enc_layer_type / --num_classes` must match the checkpoint training configuration.
- If you see `FileNotFoundError`, your trial paths likely do not match your local VoxCeleb1 directory; regenerate the trials with the correct `VOX1_ROOT`.

## Logging

- EER/minDCF are printed with 3 decimals.
- Sanity check is disabled by default to avoid long startup time when large trial sets are enabled.
- Progress bar output is disabled and `log_every_n_steps` is increased to reduce `nohup` log size.
