import json
from importlib.resources import files

import torch
import torch.nn.functional as F
import torchaudio
from datasets import Dataset as Dataset_
from datasets import load_from_disk
from torch import nn
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from f5_tts.model.modules import MelSpec
from f5_tts.model.utils import default
import librosa

QWAN_ASR_PATH = "/yangliusha03/panyuanhao/Qwen/Qwen3-ASR-1.7B"

from f5_tts.model.qwen3_audio_encoder import Qwen3ASRAudioEncoder
from transformers import WhisperFeatureExtractor

# 加载一次，全局冻结
qwen_encoder = Qwen3ASRAudioEncoder.from_qwen3_asr_pretrained(
    QWAN_ASR_PATH,
    dtype=torch.float32,
    device="cpu",  # <-- 必须 CPU
    attn_implementation="eager"
)
feature_extractor = WhisperFeatureExtractor.from_pretrained(QWAN_ASR_PATH)
qwen_encoder.eval()
for param in qwen_encoder.parameters():
    param.requires_grad = False

def extract_qwen_feat(audio_24k):
    # audio_24k: [1, T] 24kHz tensor
    audio_np = audio_24k.squeeze().cpu().numpy()
    # 1. 重采样到 16k
    audio_16k = librosa.resample(audio_np, orig_sr=24000, target_sr=16000)

    # 2. 提取特征
    feats = feature_extractor(
        audio_16k, 
        sampling_rate=16000, 
        return_tensors="pt", 
        return_attention_mask=True
    )

    feature_lens = feats["attention_mask"].sum(dim=-1)
    feat_len = int(feature_lens.item())

    input_features = feats["input_features"][0, :, :feat_len]

    with torch.no_grad():
        qwen_out = qwen_encoder(input_features, feature_lens=feature_lens, output_hidden_states=True)

    return qwen_out.hidden_states[18]  # [T, D]


class HFDataset(Dataset):
    def __init__(
        self,
        hf_dataset: Dataset,
        target_sample_rate=24_000,
        n_mel_channels=100,
        hop_length=256,
        n_fft=1024,
        win_length=1024,
        mel_spec_type="vocos",
    ):
        self.data = hf_dataset
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length

        self.mel_spectrogram = MelSpec(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
            target_sample_rate=target_sample_rate,
            mel_spec_type=mel_spec_type,
        )

    def get_frame_len(self, index):
        row = self.data[index]
        audio = row["audio"]["array"]
        sample_rate = row["audio"]["sampling_rate"]
        return audio.shape[-1] / sample_rate * self.target_sample_rate / self.hop_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]
        audio = row["audio"]["array"]

        # logger.info(f"Audio shape: {audio.shape}")

        sample_rate = row["audio"]["sampling_rate"]
        duration = audio.shape[-1] / sample_rate

        if duration > 30 or duration < 0.3:
            return self.__getitem__((index + 1) % len(self.data))

        audio_tensor = torch.from_numpy(audio).float()

        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
            audio_tensor = resampler(audio_tensor)

        audio_tensor = audio_tensor.unsqueeze(0)  # 't -> 1 t')

        mel_spec = self.mel_spectrogram(audio_tensor)

        mel_spec = mel_spec.squeeze(0)  # '1 d t -> d t'

        text = row["text"]

        return dict(
            mel_spec=mel_spec,
            text=text,
        )


class CustomDataset(Dataset):
    def __init__(
        self,
        custom_dataset: Dataset,
        durations=None,
        target_sample_rate=24_000,
        hop_length=256,
        n_mel_channels=100,
        n_fft=1024,
        win_length=1024,
        mel_spec_type="vocos",
        preprocessed_mel=False,
        mel_spec_module: nn.Module | None = None,
    ):
        self.data = custom_dataset
        self.durations = durations
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.win_length = win_length
        self.mel_spec_type = mel_spec_type
        self.preprocessed_mel = preprocessed_mel

        if not preprocessed_mel:
            self.mel_spectrogram = default(
                mel_spec_module,
                MelSpec(
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                    n_mel_channels=n_mel_channels,
                    target_sample_rate=target_sample_rate,
                    mel_spec_type=mel_spec_type,
                ),
            )

    def get_frame_len(self, index):
        if (
            self.durations is not None
        ):  # Please make sure the separately provided durations are correct, otherwise 99.99% OOM
            return self.durations[index] * self.target_sample_rate / self.hop_length
        return self.data[index]["duration"] * self.target_sample_rate / self.hop_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        while True:
            row = self.data[index]
            audio_path = row["audio_path"]
            text = row["text"]
            duration = row["duration"]
            cond_audio_path = row["cond_audio_path"]

            # filter by given length
            if 0.3 <= duration <= 30:
                break  # valid

            index = (index + 1) % len(self.data)

        # if self.preprocessed_mel:
        #     mel_spec = torch.tensor(row["mel_spec"])
        #     cond_mel_spec = torch.tensor(row["cond_mel_spec"])
        # else:
        #     audio, source_sample_rate = torchaudio.load(audio_path)

        #     # make sure mono input
        #     if audio.shape[0] > 1:
        #         audio = torch.mean(audio, dim=0, keepdim=True)
        #     # resample if necessary
        #     if source_sample_rate != self.target_sample_rate:
        #         resampler = torchaudio.transforms.Resample(source_sample_rate, self.target_sample_rate)
        #         audio = resampler(audio)
        #     # to mel spectrogram
        #     mel_spec = self.mel_spectrogram(audio)
        #     mel_spec = mel_spec.squeeze(0)  # '1 d t -> d t'


            # cond_audio, cond_sr = torchaudio.load(cond_audio_path)
            # if cond_audio.shape[0] > 1:
            #     cond_audio = torch.mean(cond_audio, dim=0, keepdim=True)
            # if cond_sr != self.target_sample_rate:
            #     resampler = torchaudio.transforms.Resample(cond_sr, self.target_sample_rate)
            #     cond_audio = resampler(cond_audio)
                
            # cond_mel_spec = self.mel_spectrogram(cond_audio)
            # cond_mel_spec = cond_mel_spec.squeeze(0)

        # return {
        #     "mel_spec": mel_spec,
        #     "cond_mel_spec": cond_mel_spec,
        #     "text": text,
        # }

        # =========================
        # 读取干净音频
        # =========================
        audio, sr = torchaudio.load(audio_path)
        if audio.shape[0] > 1:
            audio = audio.mean(0, keepdim=True)
        if sr != self.target_sample_rate:
            audio = torchaudio.transforms.Resample(sr, self.target_sample_rate)(audio)
        mel_spec = self.mel_spectrogram(audio).squeeze(0)
        # =========================
        # 🔥 读取 cond 音频 → 提取 Qwen 特征
        # =========================
        cond_audio, cond_sr = torchaudio.load(cond_audio_path)
        if cond_audio.shape[0] > 1:
            cond_audio = cond_audio.mean(0, keepdim=True)
        if cond_sr != self.target_sample_rate:
            cond_audio = torchaudio.transforms.Resample(cond_sr, self.target_sample_rate)(cond_audio)

        # 🔥 提取 Qwen 特征
        qwen_feat = extract_qwen_feat(cond_audio)

        return {
            "mel_spec": mel_spec,
            "qwen_feat": qwen_feat,  # ✅ 训练用 Qwen 特征
            "text": text,
        }


# Dynamic Batch Sampler
class DynamicBatchSampler(Sampler[list[int]]):
    """Extension of Sampler that will do the following:
    1.  Change the batch size (essentially number of sequences)
        in a batch to ensure that the total number of frames are less
        than a certain threshold.
    2.  Make sure the padding efficiency in the batch is high.
    3.  Shuffle batches each epoch while maintaining reproducibility.
    """

    def __init__(
        self, sampler: Sampler[int], frames_threshold: int, max_samples=0, random_seed=None, drop_residual: bool = False
    ):
        self.sampler = sampler
        self.frames_threshold = frames_threshold
        self.max_samples = max_samples
        self.random_seed = random_seed
        self.epoch = 0

        indices, batches = [], []
        data_source = self.sampler.data_source

        for idx in tqdm(
            self.sampler, desc="Sorting with sampler... if slow, check whether dataset is provided with duration"
        ):
            indices.append((idx, data_source.get_frame_len(idx)))
        indices.sort(key=lambda elem: elem[1])

        batch = []
        batch_frames = 0
        for idx, frame_len in tqdm(
            indices, desc=f"Creating dynamic batches with {frames_threshold} audio frames per gpu"
        ):
            if batch_frames + frame_len <= self.frames_threshold and (max_samples == 0 or len(batch) < max_samples):
                batch.append(idx)
                batch_frames += frame_len
            else:
                if len(batch) > 0:
                    batches.append(batch)
                if frame_len <= self.frames_threshold:
                    batch = [idx]
                    batch_frames = frame_len
                else:
                    batch = []
                    batch_frames = 0

        if not drop_residual and len(batch) > 0:
            batches.append(batch)

        del indices
        self.batches = batches

        # Ensure even batches with accelerate BatchSamplerShard cls under frame_per_batch setting
        self.drop_last = True

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch for this sampler."""
        self.epoch = epoch

    def __iter__(self):
        # Use both random_seed and epoch for deterministic but different shuffling per epoch
        if self.random_seed is not None:
            g = torch.Generator()
            g.manual_seed(self.random_seed + self.epoch)
            # Use PyTorch's random permutation for better reproducibility across PyTorch versions
            indices = torch.randperm(len(self.batches), generator=g).tolist()
            batches = [self.batches[i] for i in indices]
        else:
            batches = self.batches
        return iter(batches)

    def __len__(self):
        return len(self.batches)


# Load dataset


def load_dataset(
    dataset_name: str,
    tokenizer: str = "pinyin",
    dataset_type: str = "CustomDataset",
    audio_type: str = "raw",
    mel_spec_module: nn.Module | None = None,
    mel_spec_kwargs: dict = dict(),
) -> CustomDataset | HFDataset:
    """
    dataset_type    - "CustomDataset" if you want to use tokenizer name and default data path to load for train_dataset
                    - "CustomDatasetPath" if you just want to pass the full path to a preprocessed dataset without relying on tokenizer
    """

    print("Loading dataset ...")

    if dataset_type == "CustomDataset":
        rel_data_path = str(files("f5_tts").joinpath(f"../../data/{dataset_name}_{tokenizer}"))
        if audio_type == "raw":
            try:
                train_dataset = load_from_disk(f"{rel_data_path}/raw")
            except:  # noqa: E722
                train_dataset = Dataset_.from_file(f"{rel_data_path}/raw.arrow")
            preprocessed_mel = False
        elif audio_type == "mel":
            train_dataset = Dataset_.from_file(f"{rel_data_path}/mel.arrow")
            preprocessed_mel = True
        with open(f"{rel_data_path}/duration.json", "r", encoding="utf-8") as f:
            data_dict = json.load(f)
        durations = data_dict["duration"]
        train_dataset = CustomDataset(
            train_dataset,
            durations=durations,
            preprocessed_mel=preprocessed_mel,
            mel_spec_module=mel_spec_module,
            **mel_spec_kwargs,
        )

    elif dataset_type == "CustomDatasetPath":
        try:
            train_dataset = load_from_disk(f"{dataset_name}/raw")
        except:  # noqa: E722
            train_dataset = Dataset_.from_file(f"{dataset_name}/raw.arrow")

        with open(f"{dataset_name}/duration.json", "r", encoding="utf-8") as f:
            data_dict = json.load(f)
        durations = data_dict["duration"]
        train_dataset = CustomDataset(
            train_dataset, durations=durations, preprocessed_mel=preprocessed_mel, **mel_spec_kwargs
        )

    elif dataset_type == "HFDataset":
        print(
            "Should manually modify the path of huggingface dataset to your need.\n"
            + "May also the corresponding script cuz different dataset may have different format."
        )
        pre, post = dataset_name.split("_")
        train_dataset = HFDataset(
            load_dataset(f"{pre}/{pre}", split=f"train.{post}", cache_dir=str(files("f5_tts").joinpath("../../data"))),
        )

    return train_dataset


# collation


# def collate_fn(batch):
#     mel_specs = [item["mel_spec"].squeeze(0) for item in batch]
#     cond_mel_specs = [item["cond_mel_spec"].squeeze(0) for item in batch]

#     mel_lengths = torch.LongTensor([spec.shape[-1] for spec in mel_specs])
#     max_mel_length = mel_lengths.amax()

#     def pad_spec(specs, max_len):
#         padded = []
#         for spec in specs:
#             pad = (0, max_len - spec.size(-1))
#             padded.append(F.pad(spec, pad, value=0))
#         return torch.stack(padded)

#     mel_specs = pad_spec(mel_specs, max_mel_length)
#     cond_mel_specs = pad_spec(cond_mel_specs, max_mel_length)

#     text = [item["text"] for item in batch]
    
#     return {
#         "mel": mel_specs,
#         "cond_mel": cond_mel_specs,
#         "mel_lengths": mel_lengths,
#         "text": text,
#     }

def collate_fn(batch):
    mel_specs = [item["mel_spec"] for item in batch]
    qwen_feats = [item["qwen_feat"] for item in batch]

    mel_lengths = torch.LongTensor([m.shape[-1] for m in mel_specs])
    max_mel_len = mel_lengths.amax()

    def pad_mel(xs, max_len):
        out = []
        for x in xs:
            pad = (0, max_len - x.shape[-1])
            out.append(F.pad(x, pad, value=0))
        return torch.stack(out)

    def pad_qwen(xs, max_len):
        out = []
        for x in xs:
            pad = (0, 0, 0, max_len - x.shape[-2])
            out.append(F.pad(x, pad, value=0))
        return torch.stack(out)

    mel_specs = pad_mel(mel_specs, max_mel_len)
    qwen_feats = pad_qwen(qwen_feats, max_mel_len)

    return {
        "mel": mel_specs,
        "qwen_feat": qwen_feats,
        "mel_lengths": mel_lengths,
        "text": [item["text"] for item in batch]
    }