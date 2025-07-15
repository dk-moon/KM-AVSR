import os
import torch
import torchaudio
import torchvision
import soundfile as sf


def cut_or_pad(data, size, dim=0):
    """
    Pads or trims the data along a dimension.
    """
    if data.size(dim) < size:
        padding = size - data.size(dim)
        data = torch.nn.functional.pad(data, (0, 0, 0, padding), "constant")
    elif data.size(dim) > size:
        data = data[:size]
    assert data.size(dim) == size, f"Expected size {size}, but got {data.size(dim)}"
    return data


def load_video(path):
    """
    Load a video file and return it as a tensor.
    rtype: torch.Tensor, shape: T x C x H x W
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Video file not found at: {path}")
    vid = torchvision.io.read_video(path, pts_unit="sec", output_format="THWC")[0]
    if vid.numel() == 0:
        raise ValueError(f"Empty video loaded from: {path}")
    vid = vid.permute((0, 3, 1, 2))  # Convert to T x C x H x W
    # print(f"Loaded video shape: {vid.shape} from {path}")  # Debugging information
    return vid


def load_audio(path):
    """
    Load an audio file and return it as a tensor.
    rtype: torch.Tensor, shape: T x 1
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Audio file not found at: {path}")
    wav, sr = sf.read(path[:-4] + ".wav")
    if wav.ndim == 2:
        wav = wav.mean(-1)  # Convert stereo to mono
    assert sr == 16_000, f"Expected sample rate 16000, but got {sr}"
    assert len(wav.shape) == 1, "Audio should be mono"
    audio = torch.Tensor(wav).unsqueeze(1)  # Add channel dimension
    print(f"Loaded audio shape: {audio.shape} from {path}")  # Debugging information
    return audio


class AVDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir,
        label_path,
        subset,
        modality,
        audio_transform,
        video_transform,
        mouth_dir,
        wav_dir,
        rate_ratio=640,
    ):
        self.root_dir = root_dir
        self.modality = modality
        self.rate_ratio = rate_ratio
        self.list = self.load_list(label_path)
        self.audio_transform = audio_transform
        self.video_transform = video_transform
        self.mouth_dir = mouth_dir
        self.wav_dir = wav_dir

    def load_list(self, label_path):
        """
        Load the list of dataset samples from a label file.
        """
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file not found at: {label_path}")
        paths_counts_labels = []
        with open(label_path, "r") as f:
            for line in f:
                dataset_name, rel_path, input_length, token_id = line.strip().split(",")
                paths_counts_labels.append(
                    (
                        dataset_name,
                        rel_path,
                        int(input_length),
                        torch.tensor([int(x) for x in token_id.split()]),
                    )
                )
        return paths_counts_labels

    def __getitem__(self, idx):
        dataset_name, rel_path, input_length, token_id = self.list[idx]

        if self.modality == "video":
            video_path = os.path.join(self.root_dir, self.mouth_dir, rel_path)
            video = load_video(video_path)
            video = self.video_transform(video)
            return {"input": video, "target": token_id}
        elif self.modality == "audio":
            audio_path = os.path.join(self.root_dir, self.wav_dir, rel_path)
            audio = load_audio(audio_path)
            audio = self.audio_transform(audio)
            return {"input": audio, "target": token_id}
        elif self.modality == "audiovisual":
            video_path = os.path.join(self.root_dir, self.mouth_dir, rel_path)
            audio_path = os.path.join(self.root_dir, self.wav_dir, rel_path)

            video = load_video(video_path)
            audio = load_audio(audio_path)
            audio = cut_or_pad(audio, len(video) * self.rate_ratio)

            video = self.video_transform(video)
            audio = self.audio_transform(audio)
            return {"video": video, "audio": audio, "target": token_id}
        else:
            raise ValueError(f"Unsupported modality: {self.modality}")

    def __len__(self):
        return len(self.list)
