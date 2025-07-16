import os

import torch
from pytorch_lightning import LightningDataModule

from .av_dataset import AVDataset
from .samplers import (
    ByFrameCountSampler,
    DistributedSamplerWrapper,
    RandomSamplerWrapper,
)
from .transforms import AudioTransform, VideoTransform


def pad(samples, pad_val=0.0):
    lengths = [len(s) for s in samples]
    max_size = max(lengths)

    sample_shape = list(samples[0].shape[1:])
    collated_batch = samples[0].new_zeros([len(samples), max_size] + sample_shape)
    for i, sample in enumerate(samples):
        diff = len(sample) - max_size
        if diff == 0:
            collated_batch[i] = sample
        else:
            collated_batch[i] = torch.cat(
                [sample, sample.new_full([-diff] + sample_shape, pad_val)]
            )
    if len(samples[0].shape) == 1:
        collated_batch = collated_batch.unsqueeze(1)  # targets
    elif len(samples[0].shape) == 2:
        pass  # [B, T, 1]
    elif len(samples[0].shape) == 4:
        pass  # [B, T, C, H, W]
    return collated_batch, lengths


def collate_pad(batch):
    batch_out = {}
    for data_type in batch[0].keys():
        pad_val = -1 if data_type == "target" else 0.0
        c_batch, sample_lengths = pad(
            [s[data_type] for s in batch if s[data_type] is not None], pad_val
        )
        batch_out[data_type + "s"] = c_batch
        batch_out[data_type + "_lengths"] = torch.tensor(sample_lengths)
    return batch_out


class DataModule(LightningDataModule):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg
        self.cfg["gpus"] = torch.cuda.device_count()
        self.total_gpus = self.cfg["gpus"] * self.cfg.get("trainer", {}).get("num_nodes", 1)

    def _dataloader(self, ds, sampler, collate_fn):
        return torch.utils.data.DataLoader(
            ds,
            num_workers=12,
            pin_memory=True,
            batch_sampler=sampler,
            collate_fn=collate_fn,
        )

    def train_dataloader(self):
        ds_args = self.cfg["data"]["dataset"]
        train_ds = AVDataset(
            root_dir=ds_args["root_dir"],
            label_path=ds_args["train_file"],
            subset="train",
            modality=self.cfg["data"]["modality"],
            audio_transform=AudioTransform("train"),
            video_transform=VideoTransform("train"),
            mouth_dir=self.cfg["data"]["mouth_dir_train"],
            wav_dir=self.cfg["data"]["wav_dir_train"],
        )
        sampler = ByFrameCountSampler(train_ds, self.cfg["data"].get("max_frames", 300))
        if self.total_gpus > 1:
            sampler = DistributedSamplerWrapper(sampler)
        else:
            sampler = RandomSamplerWrapper(sampler)
        return self._dataloader(train_ds, sampler, collate_pad)

    def val_dataloader(self):
        ds_args = self.cfg["data"]["dataset"]
        val_ds = AVDataset(
            root_dir=ds_args["root_dir"],
            label_path=ds_args["val_file"],
            subset="val",
            modality=self.cfg["data"]["modality"],
            audio_transform=AudioTransform("val"),
            video_transform=VideoTransform("val"),
            mouth_dir=self.cfg["data"]["mouth_dir_valid"],
            wav_dir=self.cfg["data"]["wav_dir_valid"],
        )
        sampler = ByFrameCountSampler(val_ds, self.cfg["data"].get("max_frames_val", 300), shuffle=False)
        if self.total_gpus > 1:
            sampler = DistributedSamplerWrapper(sampler, shuffle=False, drop_last=True)
        return self._dataloader(val_ds, sampler, collate_pad)

    def test_dataloader(self):
        ds_args = self.cfg["data"]["dataset"]
        dataset = AVDataset(
            root_dir=ds_args["root_dir"],
            label_path=ds_args.get("test_file", ""),  # 필요 시 수정
            subset="test",
            modality=self.cfg["data"]["modality"],
            audio_transform=AudioTransform("test", snr_target=self.cfg["decode"].get("snr_target") if "decode" in self.cfg else None),
            video_transform=VideoTransform("test"),
            mouth_dir=self.cfg["data"].get("mouth_dir_test", self.cfg["data"]["mouth_dir_valid"]),
            wav_dir=self.cfg["data"].get("wav_dir_test", self.cfg["data"]["wav_dir_valid"]),
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=None)
        return dataloader