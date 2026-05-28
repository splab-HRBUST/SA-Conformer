import os
import sys
from argparse import ArgumentParser

import numpy as np
import torch
from torch.utils.data import DataLoader

# Ensure the script can be run from any working directory by adding the project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from module.dataset import Train_Dataset
from module.feature import Mel_Spectrogram
from main import Task


def extract_cohort_embeddings(
    checkpoint_path: str,
    cohort_csv_path: str,
    save_path: str,
    second: int = 3,
    batch_size: int = 128,
    num_workers: int = 8,
    device: str = "cuda",
):
    """Extract cohort embeddings from a Vox2 training CSV and save them to a .npy file.

    Args:
        checkpoint_path: Trained model checkpoint path (.ckpt), consistent with main.py
        cohort_csv_path: Vox2-only train.csv (or any CSV you want to use as cohort)
        save_path: Output .npy path, e.g., data/cohort_vox2_embeddings.npy
        second: Crop duration (seconds), keep consistent with training if needed
        batch_size: Batch size for extraction
        num_workers: DataLoader num_workers
        device: Device to use, e.g., "cuda" or "cpu"
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # 1) Restore Task via Lightning load_from_checkpoint (includes hparams)
    print(f"Loading checkpoint: {checkpoint_path}")
    model: Task = Task.load_from_checkpoint(checkpoint_path, map_location=device)
    model.to(device)
    model.eval()

    # 2) Build cohort dataset (Train_Dataset, pairs=False, aug=False)
    print(f"Loading cohort CSV: {cohort_csv_path}")
    dataset = Train_Dataset(
        train_csv_path=cohort_csv_path,
        second=second,
        pairs=False,
        aug=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    all_embs = []
    with torch.no_grad():
        for batch_idx, (waveform, labels) in enumerate(loader):
            waveform = waveform.to(device)
            feats = model.mel_trans(waveform)
            emb = model.encoder(feats)
            all_embs.append(emb.detach().cpu().numpy())
            if (batch_idx + 1) % 50 == 0:
                print(f"Processed {batch_idx + 1}/{len(loader)} batches")

    all_embs = np.concatenate(all_embs, axis=0)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, all_embs.astype(np.float32))
    print(f"Saved cohort embeddings to: {save_path}, shape={all_embs.shape}")


def main():
    parser = ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to the trained model checkpoint (.ckpt)")
    parser.add_argument("--cohort_csv_path", type=str, required=True,
                        help="Path to Vox2-only train.csv (or any CSV used as cohort)")
    parser.add_argument("--save_path", type=str, default="data/cohort_vox2_embeddings.npy",
                        help="Output .npy path")
    parser.add_argument("--second", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    extract_cohort_embeddings(
        checkpoint_path=args.checkpoint_path,
        cohort_csv_path=args.cohort_csv_path,
        save_path=args.save_path,
        second=args.second,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )


if __name__ == "__main__":
    main()
