import os
import argparse
import warnings
warnings.filterwarnings("ignore")

import torch
from pytorch_lightning import seed_everything, Trainer

from datamodule.data_module import DataModule

def main():
    parser = argparse.ArgumentParser()

    # -----------------
    # ⚙️ 공통 설정 인자
    # -----------------
    parser.add_argument("--exp_dir", type=str, default="./exp")
    parser.add_argument("--exp_name", type=str, default="eval_run")
    parser.add_argument("--modality", type=str, choices=["audio", "video", "audiovisual"], default="audio")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, choices=["cpu", "gpu"], default="cpu", help="Choose device: cpu or gpu")
    parser.add_argument("--num_devices", type=int, default=1, help="Number of GPUs to use if device=gpu")
    parser.add_argument("--pretrained_model_path", type=str, required=True, help="Path to pretrained model checkpoint")
    parser.add_argument("--label_flag", type=int, default=0, help="Print label flag during test logging")

    # backbone & optimizer-related args (필요 시 유지)
    parser.add_argument("--audiovisual_backbone", type=str, default="conformer_av")
    parser.add_argument("--audio_backbone", type=str, default="conformer_audio")
    parser.add_argument("--visual_backbone", type=str, default="conformer_visual")

    args = parser.parse_args()

    # -----------------
    # ⚙️ seed 고정
    # -----------------
    seed_everything(42, workers=True)

    # -----------------
    # ⚙️ cfg-like dict 생성
    # -----------------
    cfg = {
        "exp_dir": args.exp_dir,
        "exp_name": args.exp_name,
        "pretrained_model_path": args.pretrained_model_path,
        "label_flag": args.label_flag,
        "model": {
            "audiovisual_backbone": args.audiovisual_backbone,
            "audio_backbone": args.audio_backbone,
            "visual_backbone": args.visual_backbone,
        },
        "data": {
            "dataset": {"root_dir": "."},
            "modality": args.modality,
            "batch_size": args.batch_size,
        },
        "trainer": {
            "log_every_n_steps": 10,
        },
    }

    # -----------------
    # ⚙️ 모듈 import 분기
    # -----------------
    if args.modality in ["audio", "video"]:
        from lightning import ModelModule
    elif args.modality == "audiovisual":
        from lightning_av import ModelModule
    else:
        raise ValueError(f"Invalid modality: {args.modality}")

    modelmodule = ModelModule(cfg)
    datamodule = DataModule(cfg)

    # -----------------
    # ⚙️ 디바이스 설정
    # -----------------
    if args.device == "gpu" and torch.cuda.is_available():
        accelerator = "gpu"
        devices = args.num_devices
    else:
        accelerator = "cpu"
        devices = 1

    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        num_nodes=1,
    )

    # -----------------
    # ⚙️ 모델 로드
    # -----------------
    ckpt = torch.load(args.pretrained_model_path, map_location=lambda storage, loc: storage)

    if "state_dict" in ckpt:
        modelmodule.model.load_state_dict(ckpt["state_dict"])
    elif "model_state_dict" in ckpt:
        modelmodule.model.load_state_dict(ckpt["model_state_dict"])
    else:
        modelmodule.model.load_state_dict(ckpt)

    # -----------------
    # ⚙️ 평가 실행
    # -----------------
    trainer.test(model=modelmodule, datamodule=datamodule)


if __name__ == "__main__":
    main()