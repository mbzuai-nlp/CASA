from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DatasetConfig:
    root: str = 'data/clips'
    annotator: str = 'A1'
    sampling_rate: int = 16000
    label: str = None
    

# model configurations
@dataclass
class BaseModelConfig:
    num_labels: int = 9
    dropout: float = 0.3
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    scheduler_eta_min: float = 1e-6
    label_names: List[str] = field(default_factory=lambda: ["SR", "ISR", "MUR", "P", "B", "V", "FG", "HM", "ME"])


@dataclass
class AudioModelConfig(BaseModelConfig):
    pretrained_model_name: str = "facebook/wav2vec2-base-960h"
    freeze_encoder: bool = True
    freeze_feature_extractor: bool = True
    unfreeze_layers: Optional[List[int]] = None
    scheduler_warmup_steps: int = 0
    loss_weights: List[float] = field(default_factory=lambda: [
        # 0.9,  # SR
        # 0.7,  # ISR
        # 0.9,  # MUR
        # 0.9,  # P
        # 0.6,  # B
        1.0,  # any
    ])


@dataclass
class VideoModelConfig(BaseModelConfig):
    pretrained_model_name: str = "google/vivit-b-16x2-kinetics400"
    freeze_backbone: bool = True
    unfreeze_last_n_layers: int = 2
    learning_rate: float = 1e-5  # Override with lower default rate for video
    loss_weights: List[float] = field(default_factory=lambda: [
        # 0.9,  # SR
        # 0.7,  # ISR
        # 0.9,  # MUR
        # 0.9,  # P
        # 0.6,  # B
        1.0,  # any
    ])


@dataclass
class MultimodalModelConfig(BaseModelConfig):
    audio_config: AudioModelConfig = AudioModelConfig()
    video_config: VideoModelConfig = VideoModelConfig()
    fusion_method: str = "concat"
    fusion_dim: int = 512  # Default fusion dimension, can be adjusted based on model output sizes
    loss_weights: List[float] = field(default_factory=lambda: [
        # 0.9,  # SR
        # 0.7,  # ISR
        # 0.9,  # MUR
        # 0.9,  # P
        # 0.6,  # B
        1.0,  # any
    ])

@dataclass
class TrainingConfig:
    seed: int = 42
    batch_size: int = 32
    num_workers: int = 4
    max_epochs: int = 10
    gradient_clip_val: float = 1.0
    modality: str = "audio"

    # dataset configurations
    dataset_config: DatasetConfig = DatasetConfig()

    # model configurations
    audio_model_config: AudioModelConfig = AudioModelConfig()
    video_model_config: VideoModelConfig = VideoModelConfig()
    fusion_method: str = "concat"
    fusion_dim: int = 512

    # output configurations
    output_dir: str = "output"

        
# update configs from args
def from_args(args):
    config = TrainingConfig()
    for key, value in vars(args).items():
        if hasattr(config, key):
            setattr(config, key, value)
        elif 'audio' in key:
            audio_key = key.replace('audio_', '')
            if hasattr(config.audio_model_config, audio_key):
                setattr(config.audio_model_config, audio_key, value)
        elif 'video' in key:
            video_key = key.replace('video_', '')
            if hasattr(config.video_model_config, video_key):
                setattr(config.video_model_config, video_key, value)
        elif 'dataset' in key:
            dataset_key = key.replace('dataset_', '')
            if hasattr(config.dataset_config, dataset_key):
                setattr(config.dataset_config, dataset_key, value)
        else:
            print(f"Warning: {key} not found in config")
    return config
