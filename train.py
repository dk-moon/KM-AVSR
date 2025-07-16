import os
import argparse
import warnings
warnings.filterwarnings("ignore")
import torch
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from datamodule.data_module import DataModule
from utils.avg_ckpts import ensemble

def main():
    parser = argparse.ArgumentParser()

    # ⚙️ 공통 실험 설정
    parser.add_argument("--exp_dir", type=str, default="./exp")
    parser.add_argument("--exp_name", type=str, default="test_run")
    parser.add_argument("--modality", type=str, choices=["audio", "video", "audiovisual"], default="audiovisual")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--device", type=str, choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--num_devices", type=int, default=1)

    # ⚙️ 데이터셋
    parser.add_argument("--dataset_root", type=str, default="dataset")
    parser.add_argument("--train_file", type=str, default="dataset/train/train.csv")
    parser.add_argument("--val_file", type=str, default="dataset/valid/valid.csv")
    parser.add_argument("--mouth_dir_train", type=str, default="dataset/train/Video_npy")
    parser.add_argument("--wav_dir_train", type=str, default="dataset/train/Audio")
    parser.add_argument("--mouth_dir_valid", type=str, default="dataset/valid/Video_npy")
    parser.add_argument("--wav_dir_valid", type=str, default="dataset/valid/Audio")

    # ⚙️ Backbone 설정 추가 ✅
    parser.add_argument("--audiovisual_backbone", type=str, default="resnet_conformer")
    parser.add_argument("--audio_backbone", type=str, default="resnet_conformer")
    parser.add_argument("--visual_backbone", type=str, default="resnet_conformer")

    # ⚙️ 모델 설정
    parser.add_argument("--pretrained_model_path", type=str, default=None)
    parser.add_argument("--transfer_frontend", action="store_true")
    parser.add_argument("--transfer_encoder", action="store_true")
    parser.add_argument("--label_flag", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=5)

    # ⚙️ Trainer 고급 설정
    parser.add_argument("--precision", type=int, default=32)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--sync_batchnorm", action="store_true")
    parser.add_argument("--num_sanity_val_steps", type=int, default=0)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--gradient_clip_val", type=float, default=5.0)

    args = parser.parse_args()

    # Seed & GPU 설정
    seed_everything(42, workers=True)

    # Checkpoint & Callbacks
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

    # ⚙️ 데이터 모듈 & 모델 모듈
    if args.modality in ["audio", "video"]:
        from lightning_singlemodal import ModelModule
    elif args.modality == "audiovisual":
        from lightning_multimodal import ModelModule
    else:
        raise ValueError(f"Invalid modality: {args.modality}")

    # Backbone config dict
    backbone_config = {
        "adim": 768,
        "aheads": 12,
        "eunits": 3072,
        "elayers": 12,
        "transformer_input_layer": "conv3d" if args.modality in ["video", "audiovisual"] else "conv1d",
        "dropout_rate": 0.1,
        "transformer_attn_dropout_rate": 0.1,
        "transformer_encoder_attn_layer_type": "rel_mha",
        "macaron_style": True,
        "use_cnn_module": True,
        "cnn_module_kernel": 31,
        "zero_triu": False,
        "a_upsample_ratio": 1,
        "relu_type": "swish",
        "ddim": 768,
        "dheads": 12,
        "dunits": 3072,
        "dlayers": 6,
        "lsm_weight": 0.1,
        "transformer_length_normalized_loss": False,
        "mtlalpha": 0.1,
        "ctc_type": "builtin",
        "rel_pos_type": "latest",
        "aux_adim": 768,
        "aux_aheads": 12,
        "aux_eunits": 3072,
        "aux_elayers": 12,
        "aux_transformer_input_layer": "conv1d",  # Audio encoder uses conv1d
        "aux_dropout_rate": 0.1,
        "aux_transformer_attn_dropout_rate": 0.1,
        "aux_transformer_encoder_attn_layer_type": "rel_mha",
        "aux_macaron_style": True,
        "aux_use_cnn_module": True,
        "aux_cnn_module_kernel": 31,
        "aux_zero_triu": False,
        "aux_a_upsample_ratio": 1,
        "aux_relu_type": "swish",
        "fusion_hdim": 768,
        "fusion_norm": None,
    }

    cfg = {
        "exp_dir": args.exp_dir,
        "exp_name": args.exp_name,
        "pretrained_model_path": args.pretrained_model_path,
        "transfer_frontend": args.transfer_frontend,
        "transfer_encoder": args.transfer_encoder,
        "label_flag": args.label_flag,
        "model": {
            "audiovisual_backbone": backbone_config,
            "audio_backbone": args.audio_backbone,
            "visual_backbone": args.visual_backbone,
        },
        "optimizer": {
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "warmup_epochs": args.warmup_epochs,
        },
        "data": {
            "dataset": {
                "root_dir": args.dataset_root,
                "train_file": args.train_file,
                "val_file": args.val_file,
            },
            "modality": args.modality,
            "batch_size": args.batch_size,
            "mouth_dir_train": args.mouth_dir_train,
            "wav_dir_train": args.wav_dir_train,
            "mouth_dir_valid": args.mouth_dir_valid,
            "wav_dir_valid": args.wav_dir_valid,
        },
        "trainer": {
            "max_epochs": args.max_epochs,
            "log_every_n_steps": 10,
        },
    }

    modelmodule = ModelModule(cfg)
    datamodule = DataModule(cfg)

    # Device 설정
    if args.device == "gpu" and torch.cuda.is_available():
        accelerator = "gpu"
        devices = args.num_devices
    else:
        accelerator = "cpu"
        devices = 1

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
        callbacks=callbacks,
        use_distributed_sampler=False,
        default_root_dir=args.exp_dir,
    )

    trainer.fit(model=modelmodule, datamodule=datamodule)
    ensemble(cfg)

if __name__ == "__main__":
    main()