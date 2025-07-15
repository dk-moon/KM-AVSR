import os
import argparse
import logging

import torch
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

from datamodule.data_module import DataModule
from utils.avg_ckpts import ensemble

def main():
    parser = argparse.ArgumentParser()

    # -----------------
    # ⚙️ 공통 실험 설정 인자
    # -----------------
    parser.add_argument("--exp_dir", type=str, default="./exp")
    parser.add_argument("--exp_name", type=str, default="test_run")
    parser.add_argument("--modality", type=str, choices=["audio", "video", "audiovisual"], default="audio")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--device", type=str, choices=["cpu", "gpu"], default="cpu", help="Choose device: cpu or gpu")
    parser.add_argument("--num_devices", type=int, default=1, help="Number of GPUs to use if device=gpu")

    # -----------------
    # ⚙️ lightning 관련 추가 인자
    # -----------------
    parser.add_argument("--pretrained_model_path", type=str, default=None, help="Path to pretrained model checkpoint")
    parser.add_argument("--transfer_frontend", action="store_true", help="Transfer only frontend layers from checkpoint")
    parser.add_argument("--transfer_encoder", action="store_true", help="Transfer only encoder layers from checkpoint")
    parser.add_argument("--label_flag", type=int, default=0, help="Print label flag during test logging")
    parser.add_argument("--audio_backbone", type=str, default="conformer_audio", help="Backbone name for audio model")
    parser.add_argument("--visual_backbone", type=str, default="conformer_visual", help="Backbone name for visual model")
    parser.add_argument("--audiovisual_backbone", type=str, default="conformer_av", help="Backbone name for audiovisual model")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=5)

    # -----------------
    # ⚙️ Trainer 고급 인자
    # -----------------
    parser.add_argument("--precision", type=int, default=32, help="Precision, e.g., 32 or 16 for mixed precision")
    parser.add_argument("--num_nodes", type=int, default=1, help="Number of nodes for distributed training")
    parser.add_argument("--sync_batchnorm", action="store_true", help="Enable synchronized BatchNorm across GPUs")
    parser.add_argument("--num_sanity_val_steps", type=int, default=0, help="Number of sanity validation steps")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--gradient_clip_val", type=float, default=5.0, help="Gradient clipping value")
    parser.add_argument("--replace_sampler_ddp", action="store_false", help="Replace sampler in DDP (default True)")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume from")

    args = parser.parse_args()

    # -----------------
    # ⚙️ seed & GPU 설정
    # -----------------
    seed_everything(42, workers=True)

    # -----------------
    # ⚙️ PyTorch Lightning Trainer config
    # -----------------
    checkpoint = ModelCheckpoint(
        monitor="monitoring_step",
        mode="max",
        dirpath=os.path.join(args.exp_dir, args.exp_name) if args.exp_dir else None,
        save_last=True,
        filename="{epoch}",
        save_top_k=10,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint, lr_monitor]

    # -----------------
    # ⚙️ 데이터 모듈 & 모델 모듈
    # -----------------
    if args.modality in ["audio", "video"]:
        from lightning_singlemodal import ModelModule
    elif args.modality == "audiovisual":
        from lightning_multimodal import ModelModule
    else:
        raise ValueError(f"Invalid modality: {args.modality}")

    # cfg-like dict 생성
    cfg = {
        "exp_dir": args.exp_dir,
        "exp_name": args.exp_name,
        "pretrained_model_path": args.pretrained_model_path,
        "transfer_frontend": args.transfer_frontend,
        "transfer_encoder": args.transfer_encoder,
        "label_flag": args.label_flag,
        "model": {
            "audiovisual_backbone": args.audiovisual_backbone,
            "audio_backbone": args.audio_backbone,
            "visual_backbone": args.visual_backbone,
        },
        "optimizer": {
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "warmup_epochs": args.warmup_epochs,
        },
        "data": {
            "dataset": {"root_dir": "."},
            "modality": args.modality,
            "batch_size": args.batch_size,
        },
        "trainer": {
            "max_epochs": args.max_epochs,
            "log_every_n_steps": 10,
        },
    }

    modelmodule = ModelModule(cfg)
    datamodule = DataModule(cfg)

    # -----------------
    # ⚙️ Trainer device 설정
    # -----------------
    if args.device == "gpu" and torch.cuda.is_available():
        accelerator = "gpu"
        devices = args.num_devices
    else:
        accelerator = "cpu"
        devices = 1

    # -----------------
    # ⚙️ Trainer 생성
    # -----------------
    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator=accelerator,
        devices=devices,
        precision=args.precision,
        num_nodes=args.num_nodes,
        sync_batchnorm=args.sync_batchnorm,
        num_sanity_val_steps=args.num_sanity_val_steps,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        replace_sampler_ddp=args.replace_sampler_ddp,
        callbacks=callbacks,
        use_distributed_sampler=False,
        default_root_dir=args.exp_dir,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )

    trainer.fit(model=modelmodule, datamodule=datamodule)
    ensemble(cfg)


if __name__ == "__main__":
    main()