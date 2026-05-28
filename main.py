from argparse import ArgumentParser
from copy import deepcopy
from typing import Any, Union
import torch.distributed as dist
from pytorch_lightning.strategies import DDPStrategy
import random

import torch
import torch.nn as nn
import numpy as np
import os

from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR, CyclicLR

from module.feature import Mel_Spectrogram
from module.loader import SPK_datamodule
import score as score
from loss import softmax, amsoftmax
from loss.circle_loss import SimpleCircleLoss

class Task(LightningModule):
    def __init__(
        self,
        learning_rate: float = 0.2,
        weight_decay: float = 1.5e-6,
        batch_size: int = 32,
        num_workers: int = 10,
        max_epochs: int = 1000,
        trial_path: str = "data/vox1_test.txt",
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        trial_paths = getattr(self.hparams, "trial_paths", None)
        if trial_paths is None:
            trial_paths = [getattr(self.hparams, "trial_path", None)]

        self.trial_paths = [p for p in list(trial_paths) if p]
        if len(self.trial_paths) == 0:
            raise ValueError("No valid trial paths provided. Use --trial_paths or --trial_path.")

        self.trials_dict = {}
        for p in self.trial_paths:
            name = os.path.splitext(os.path.basename(p))[0]
            self.trials_dict[name] = np.loadtxt(p, str)

        self.primary_trial_name = next(iter(self.trials_dict.keys()))
        self.trials = self.trials_dict[self.primary_trial_name]

        self.mel_trans = Mel_Spectrogram()

        from module.resnet import resnet34, resnet18, resnet34_large
        from module.ecapa_tdnn import ecapa_tdnn, ecapa_tdnn_large
        from module.transformer_cat import transformer_cat
        from module.conformer import conformer
        from module.conformer_cat import conformer_cat
        from module.conformer_weight import conformer_weight

        if self.hparams.encoder_name == "resnet18":
            self.encoder = resnet18(embedding_dim=self.hparams.embedding_dim)

        elif self.hparams.encoder_name == "resnet34":
            self.encoder = resnet34_large(embedding_dim=self.hparams.embedding_dim)

        elif self.hparams.encoder_name == "ecapa_tdnn":
            self.encoder = ecapa_tdnn(embedding_dim=self.hparams.embedding_dim)

        elif self.hparams.encoder_name == "ecapa_tdnn_large":
            self.encoder = ecapa_tdnn_large(embedding_dim=self.hparams.embedding_dim)

        elif self.hparams.encoder_name == "conformer":
            print("num_blocks is {}".format(self.hparams.num_blocks))
            self.encoder = conformer(embedding_dim=self.hparams.embedding_dim, 
                    num_blocks=self.hparams.num_blocks, input_layer=self.hparams.input_layer)

        elif self.hparams.encoder_name == "transformer_cat":
            print("num_blocks is {}".format(self.hparams.num_blocks))
            self.encoder = transformer_cat(embedding_dim=self.hparams.embedding_dim, 
                    num_blocks=self.hparams.num_blocks, input_layer=self.hparams.input_layer)

        elif self.hparams.encoder_name == "conformer_cat":
            print("num_blocks is {}".format(self.hparams.num_blocks))
            self.encoder = conformer_cat(embedding_dim=self.hparams.embedding_dim, 
                    num_blocks=self.hparams.num_blocks, input_layer=self.hparams.input_layer,
                    pos_enc_layer_type=self.hparams.pos_enc_layer_type)

        elif self.hparams.encoder_name == "conformer_weight":
            print("num_blocks is {}".format(self.hparams.num_blocks))
            self.encoder = conformer_weight(embedding_dim=self.hparams.embedding_dim, 
                    num_blocks=self.hparams.num_blocks, input_layer=self.hparams.input_layer)

        else:
            raise ValueError("encoder name error")

        if self.hparams.loss_name == "amsoftmax":
            margin = getattr(self.hparams, "am_margin", 0.2)
            scale = getattr(self.hparams, "am_scale", 30.0)
            self.loss_fun = amsoftmax(
                embedding_dim=self.hparams.embedding_dim,
                num_classes=self.hparams.num_classes,
                margin=margin,
                scale=scale,
            )
        elif self.hparams.loss_name == "circle":
            circle_margin = getattr(self.hparams, "circle_margin", 0.25)
            circle_scale = getattr(self.hparams, "circle_scale", 256.0)
            self.loss_fun = SimpleCircleLoss(
                embedding_dim=self.hparams.embedding_dim,
                num_classes=self.hparams.num_classes,
                scale=circle_scale,
                margin=circle_margin,
            )
        else:
            self.loss_fun = softmax(
                embedding_dim=self.hparams.embedding_dim,
                num_classes=self.hparams.num_classes,
            )

    def forward(self, x):
        feature = self.mel_trans(x)
        embedding = self.encoder(feature)
        return embedding

    def training_step(self, batch, batch_idx):
        waveform, label = batch
        feature = self.mel_trans(waveform)
        embedding = self.encoder(feature)
        loss, acc = self.loss_fun(embedding, label)
        self.log('train_loss', loss, prog_bar=True)
        self.log('acc', acc, prog_bar=True)
        return loss

    def on_test_epoch_start(self):
        return self.on_validation_epoch_start()

    def on_validation_epoch_start(self):
        self.index_mapping = {}
        self.eval_vectors = []

    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        x, path = batch
        path = path[0]
        with torch.no_grad():
            x = self.mel_trans(x)
            self.encoder.eval()
            x = self.encoder(x)
        x = x.detach().cpu().numpy()[0]
        self.eval_vectors.append(x)
        self.index_mapping[path] = batch_idx

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs)

    def validation_epoch_end(self, outputs):
        # num_gpus = torch.cuda.device_count()
        # eval_vectors = [None for _ in range(num_gpus)]
        # dist.all_gather_object(eval_vectors, self.eval_vectors)
        # eval_vectors = np.vstack(eval_vectors)

        # table = [None for _ in range(num_gpus)]
        # dist.all_gather_object(table, self.index_mapping)

        # Single-GPU evaluation
        eval_vectors = np.vstack(self.eval_vectors)
        index_mapping = self.index_mapping

        # index_mapping = {}
        # for i in table:
        #     index_mapping.update(i)

        eval_vectors = eval_vectors - np.mean(eval_vectors, axis=0)

        epoch = int(getattr(self, "current_epoch", 0))
        warmup = int(getattr(self.hparams, "eval_warmup_epochs", 0))

        if warmup > 0 and epoch < warmup:
            items = [(self.primary_trial_name, self.trials_dict[self.primary_trial_name])]
        else:
            items = list(self.trials_dict.items())

        for name, trials in items:
            labels, scores = score.cosine_score(trials, index_mapping, eval_vectors)
            EER, threshold = score.compute_eer(labels, scores)

            prefix = "" if name == self.primary_trial_name else (name + "_")

            print("\n[{}] cosine EER: {:.3f}% with threshold {:.3f}".format(name, EER * 100, threshold))
            self.log(prefix + "cosine_eer", EER * 100)

            minDCF, threshold = score.compute_minDCF(labels, scores, p_target=0.01)
            print("[{}] cosine minDCF(10-2): {:.3f} with threshold {:.3f}".format(name, minDCF, threshold))
            self.log(prefix + "cosine_minDCF(10-2)", minDCF)

            minDCF, threshold = score.compute_minDCF(labels, scores, p_target=0.001)
            print("[{}] cosine minDCF(10-3): {:.3f} with threshold {:.3f}".format(name, minDCF, threshold))
            self.log(prefix + "cosine_minDCF(10-3)", minDCF)

            if hasattr(self.hparams, 'use_asnorm') and self.hparams.use_asnorm:
                if getattr(self.hparams, "cohort_embedding_path", None) is None:
                    print("Warning: use_asnorm=True but cohort_embedding_path is not provided. Skipping AS-Norm.")
                else:
                    if not os.path.exists(self.hparams.cohort_embedding_path):
                        print("Warning: cohort_embedding_path not found: {}. Skipping AS-Norm.".format(self.hparams.cohort_embedding_path))
                    else:
                        cohort_vectors = np.load(self.hparams.cohort_embedding_path)
                        labels_asnorm, scores_asnorm = score.as_norm_score(
                            trials, index_mapping, eval_vectors,
                            cohort_vectors=cohort_vectors,
                            top_n=self.hparams.asnorm_top_n,
                        )
                        EER_asnorm, threshold_asnorm = score.compute_eer(labels_asnorm, scores_asnorm)
                        print("[{}] AS-Norm EER: {:.3f}% with threshold {:.3f}".format(name, EER_asnorm * 100, threshold_asnorm))
                        self.log(prefix + "asnorm_eer", EER_asnorm * 100)

        # Print semantic fusion / learnable grouping statistics once per validation epoch
        if hasattr(self.encoder, "get_fusion_info"):
            try:
                epoch = getattr(self, "current_epoch", None)
                if epoch is None and self.trainer is not None:
                    epoch = self.trainer.current_epoch
                info = self.encoder.get_fusion_info()
                print("\n[Epoch {}] Semantic fusion & grouping stats:\n{}\n".format(epoch, info))
            except Exception as e:
                print("[Epoch {}] get_fusion_info failed: {}".format(
                    getattr(self, "current_epoch", "?"), e))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        scheduler = StepLR(optimizer, step_size=self.hparams.step_size, gamma=self.hparams.gamma)
        return [optimizer], [scheduler]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                       optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
        # warm up learning_rate
        if self.trainer.global_step < self.hparams.warmup_step:
            lr_scale = min(1., float(self.trainer.global_step +
                           1) / float(self.hparams.warmup_step))
            for idx, pg in enumerate(optimizer.param_groups):
                pg['lr'] = lr_scale * self.hparams.learning_rate
        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        (args, _) = parser.parse_known_args()

        parser.add_argument("--num_workers", default=40, type=int)
        parser.add_argument("--embedding_dim", default=256, type=int)
        parser.add_argument("--num_classes", type=int, default=1211)
        parser.add_argument("--num_blocks", type=int, default=6)

        parser.add_argument("--input_layer", type=str, default="conv2d")
        parser.add_argument("--pos_enc_layer_type", type=str, default="abs_pos")

        parser.add_argument("--second", type=int, default=3)
        parser.add_argument('--step_size', type=int, default=1)
        parser.add_argument('--gamma', type=float, default=0.9)
        parser.add_argument("--batch_size", type=int, default=80)

        parser.add_argument("--learning_rate", type=float, default=0.0005)
        parser.add_argument("--warmup_step", type=float, default=4000)
        parser.add_argument("--weight_decay", type=float, default=0.000001)

        parser.add_argument("--save_dir", type=str, default=None)
        parser.add_argument("--checkpoint_path", type=str, default=None)
        parser.add_argument("--resume_ckpt_path", type=str, default=None)  # resume training from checkpoint

        # Loss hyperparameters
        parser.add_argument("--loss_name",type=str,default="amsoftmax",choices=["amsoftmax", "softmax", "circle"],)
        parser.add_argument("--am_margin", type=float, default=0.2)
        parser.add_argument("--am_scale", type=float, default=30.0)
        parser.add_argument("--circle_margin", type=float, default=0.25)
        parser.add_argument("--circle_scale", type=float, default=256.0)

        parser.add_argument("--encoder_name", type=str, default="resnet34")

        parser.add_argument("--train_csv_path", type=str, default="data/train.csv")
        parser.add_argument(
            "--trial_path",
            type=str,
            nargs="?",
            default="data/vox1_test.txt",
        )
        parser.add_argument(
            "--trial_paths",
            type=str,
            nargs="+",
            default=None,
            help="Evaluate multiple trials simultaneously; if provided, overrides --trial_path. Example: --trial_paths data/vox1_test.txt data/vox1_E_trials.txt data/vox1_H_trials.txt",
        )
        parser.add_argument(
            "--eval_warmup_epochs",
            type=int,
            default=5,
            help="For the first N epochs, evaluate only the 1st entry in trial_paths (typically VoxCeleb1-O). Starting from epoch N, evaluate all trials.",
        )
        parser.add_argument("--score_save_path", type=str, default=None)

        # 打分后处理相关超参
        parser.add_argument(
            "--use_asnorm",
            action="store_true",
            help="Use AS-Norm to normalize cosine scores during validation/testing",
        )
        parser.add_argument(
            "--asnorm_top_n",
            type=int,
            default=200,
            help="Number of cohort samples selected per utterance in AS-Norm",
        )
        parser.add_argument(
            "--cohort_embedding_path",
            type=str,
            default=None,
            help="预先提取好的 cohort embedding 的 .npy 路径（shape=[N_cohort, embedding_dim]）",
        )

        parser.add_argument('--eval', action='store_true')
        parser.add_argument('--aug', action='store_true')


        parser.add_argument(
            "--seed",
            type=int,
            default=None,
            help="全局随机种子（Python/NumPy/PyTorch/DataLoader worker）。不传则随机生成，并写入 save_dir/seed.txt",
        )
        parser.add_argument(
            "--deterministic",
            action="store_true",
            help="开启后强制 cudnn.deterministic=True、benchmark=False，可复现性更强但训练可能略慢",
        )
        return parser


def cli_main():
    parser = ArgumentParser()
    # trainer args
    parser = Trainer.add_argparse_args(parser)

    # model args
    parser = Task.add_model_specific_args(parser)
    args = parser.parse_args()

    if getattr(args, "trial_path", None) is None:
        args.trial_path = "data/vox1_test.txt"

    assert args.save_dir is not None
    os.makedirs(args.save_dir, exist_ok=True)

    seed = getattr(args, "seed", None)
    if seed is None:
        seed = int.from_bytes(os.urandom(4), byteorder="little", signed=False)
        args.seed = seed

    seed_everything(seed, workers=True)
    print("随机种子: seed={}, deterministic={}".format(seed, getattr(args, "deterministic", False)))

    try:
        with open(os.path.join(args.save_dir, "seed.txt"), "w", encoding="utf-8") as f:
            f.write(str(seed) + "\n")
    except OSError as e:
        print("警告: 无法写入 seed.txt: {}".format(e))

    if getattr(args, "deterministic", False):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except (TypeError, AttributeError):
            pass

    model = Task(**args.__dict__)

    # 加载预训练权重
    if args.checkpoint_path is not None and args.resume_ckpt_path is None:
        state_dict = torch.load(args.checkpoint_path, map_location="cpu")["state_dict"]
        model.load_state_dict(state_dict, strict=True)
        print("load weight from {}".format(args.checkpoint_path))

    checkpoint_callback = ModelCheckpoint(
        monitor='cosine_eer', 
        save_top_k=100,
        filename="{epoch}_{cosine_eer:.2f}", 
        dirpath=args.save_dir,
        every_n_epochs=1
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # init default datamodule
    print("data augmentation {}".format(args.aug))
    trial_paths = getattr(args, "trial_paths", None)
    if trial_paths is None:
        trial_paths = [args.trial_path]

    dm = SPK_datamodule(
        train_csv_path=args.train_csv_path,
        trial_paths=trial_paths,
        eval_warmup_epochs=getattr(args, "eval_warmup_epochs", 0),
        second=args.second,
        aug=args.aug,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pairs=False,
    )
    
    # 确定检查点路径
    ckpt_path = None
    if hasattr(args, 'resume_ckpt_path') and args.resume_ckpt_path is not None:
        ckpt_path = args.resume_ckpt_path
        print("恢复训练，从检查点: {}".format(ckpt_path))
        # 验证检查点文件是否存在
        if not os.path.exists(ckpt_path):
            print(f"警告: 检查点文件不存在: {ckpt_path}")
            ckpt_path = None

    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices=1,
        strategy=None,
        num_sanity_val_steps=0,
        sync_batchnorm=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[checkpoint_callback, lr_monitor],
        default_root_dir=args.save_dir,
        reload_dataloaders_every_n_epochs=1,
        accumulate_grad_batches=1,
        log_every_n_steps=200,
        deterministic=getattr(args, "deterministic", False),
    )
    
    if args.eval:
        trainer.test(model, datamodule=dm)
    else:
        # 关键：在这里传入ckpt_path参数
        trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)


if __name__ == "__main__":
    cli_main()

