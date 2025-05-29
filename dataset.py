import os
import torch
import pandas as pd
import torchaudio
import torchvision
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Union
from transformers import Wav2Vec2FeatureExtractor, VivitImageProcessor, VivitForVideoClassification
from configs import TrainingConfig, DatasetConfig

from sklearn.utils import resample


def _process_audio(audio_path: str, processor) -> Dict[str, torch.Tensor]:
    """Process audio file using Wav2Vec2 processor"""
    audio, sr = torchaudio.load(audio_path)
    
    # Resample if needed
    if sr != processor.sampling_rate:
        resampler = torchaudio.transforms.Resample(sr, processor.sampling_rate)
        audio = resampler(audio)
    
    # Convert to mono if stereo
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
        
    # Process with Wav2Vec2 processor
    processed = processor(
        audio.numpy(),
        sampling_rate=processor.sampling_rate,
        return_tensors="pt"
    )
    
    return processed

def _process_video(video_path: str, processor) -> Dict[str, torch.Tensor]:
    """Process video frames using Vivit processor"""
    try:
        # Read video frames
        video_frames, _, _ = torchvision.io.read_video(video_path, pts_unit='sec')  # [T, H, W, C]
        
        # Handle videos with too few frames
        if video_frames.shape[0] < 16:
            # Duplicate last frame to reach 16 frames
            pad_frames = 16 - video_frames.shape[0]
            last_frame = video_frames[-1:].repeat(pad_frames, 1, 1, 1)
            video_frames = torch.cat([video_frames, last_frame], dim=0)
        
        # Uniformly sample 16 frames if more frames exist
        if video_frames.shape[0] > 16:
            indices = torch.linspace(0, video_frames.shape[0] - 1, 16).long()
            video_frames = video_frames[indices]
        
        # Convert to proper format
        video_frames = video_frames.permute(0, 3, 1, 2)  # [T, C, H, W]
        
        # Convert frames to list of PIL images
        frames = [torchvision.transforms.ToPILImage()(frame) for frame in video_frames]
        
        # Process with Vivit processor
        processed = processor(
            images=frames,
            return_tensors="pt",
        )
        
        return processed
    except Exception as e:
        print(f"Error processing video {video_path}: {e}")
        return {"pixel_values": torch.zeros(1, 16, 3, 224, 224)}

def process_label(clip_data: pd.Series, labels: List[str]) -> Dict[str, Optional[int]]:
    """Process labels from dataframe row"""
    label_dict = {}
    for label in labels:
        if label in clip_data and not pd.isna(clip_data[label]):
            label_dict[label] = int(clip_data[label])
        else:
            label_dict[label] = None
    label_dict['any'] = any(val is not None for val in label_dict.values())
    return label_dict

class BaseDataset(Dataset):
    """Base dataset class with common functionality for all modalities"""
    
    labels = "SR,ISR,MUR,P,B,V,FG,HM,ME".split(",")
    primary_labels = "SR,ISR,MUR,P,B".split(",")
    secondary_labels = "V,FG,HM,ME".split(",")
    
    def __init__(self, 
                 root: str, 
                 label_path: str,
                 gold_label_path: Optional[str] = None,
                 use_gold: bool = False, 
                 feature_dir: Optional[str] = None,
                 precompute: bool = True,
                 split: str = "train",
                 label: Optional[List[str]] = None):
        """
        Initialize base dataset
        
        Args:
            root: Root directory containing data
            label_path: Path to CSV file with labels
            feature_dir: Directory to store/load precomputed features
            precompute: Whether to precompute features
        """
        self.root = root
        self.feature_dir = feature_dir
        self.label_path = label_path
        self.label = label if label else self.labels[0]
        self.split = split

        self.data_df = self.prep_df()
        self.clips = self.data_df['clip_id']
        
        # Create feature directory if needed
        if feature_dir and precompute:
            os.makedirs(f"{self.root}/{feature_dir}", exist_ok=True)
            self._precompute_features()
    
    def prep_df(self):
        df = pd.read_csv(self.label_path)
        df = df[df['split'] == self.split].reset_index(drop=True)

        if self.split == "test":
            return df
        
        df_majority = df[df[self.label]==0]
        df_minority = df[df[self.label]==1]

        df_minority_upsampled = resample(df_minority,
                                            replace=True,    
                                            n_samples=len(df_majority),    
                                            random_state=42)
        df_upsampled = pd.concat([df_majority, df_minority_upsampled])
        df_upsampled = df_upsampled.sample(frac=1, random_state=42).reset_index(drop=True)

        return df_upsampled

    def process_label(self, clip_data):
        """Process labels from dataframe row"""
        labels = {}
        for label in self.labels:
            if label in clip_data and not pd.isna(clip_data[label]):
                labels[label] = clip_data[label]
            else:
                labels[label] = None
        labels['any'] = any(val for val in labels.values() if val is not None)
        # labels['primary'] = any(val for val in labels.values() if val in self.primary_labels)
        # labels['secondary'] = any(val for val in labels.values() if val in self.secondary_labels)
        return labels
    
    def _precompute_features(self):
        """Abstract method to be implemented by child classes"""
        raise NotImplementedError
    
    def __len__(self) -> int:
        return len(self.clips)
    
    def get_item_paths(self, clip_id):
        """Get paths for a specific clip ID"""
        return {
            'audio': f"{self.root}/audios/{clip_id}.wav",
            'video': f"{self.root}/videos/{clip_id}.mp4",
            'audio_features': f"{self.root}/{self.feature_dir}/{clip_id}_audio.pt",
            'video_features': f"{self.root}/{self.feature_dir}/{clip_id}_video.pt",
        }

class AudioDataset(BaseDataset):
    """Dataset for audio-only processing using Wav2Vec2"""
    
    def __init__(self, 
                 root: str, 
                 label_path: str, 
                 gold_label_path: Optional[str] = None,
                 use_gold: bool = False,
                 feature_dir: str = 'features',
                 precompute: bool = True,
                 sampling_rate: int = 16000,
                 split: str = "train",
                 label: Optional[List[str]] = None,
                 **kwargs):
        
        # Initialize audio processor before calling parent constructor
        self.audio_processor = Wav2Vec2FeatureExtractor.from_pretrained(
            "facebook/wav2vec2-base-960h",
            sampling_rate=sampling_rate
        )
        
        self.target_sampling_rate = sampling_rate
        
        super().__init__(root, label_path, gold_label_path, use_gold, feature_dir, precompute, split, label)
    
    def process_audio(self, audio_path: str) -> Dict[str, torch.Tensor]:
        return _process_audio(audio_path, self.audio_processor)
    
    def _precompute_features(self):
        """Precompute audio features for all clips"""
        print(f"Precomputing audio features for {len(self)} clips...")
        for idx in range(len(self)):
            clip_id = self.clips[idx]
            audio_path = self.get_item_paths(clip_id)['audio']
            
            # Skip if already computed
            feature_path = f"{self.root}/{self.feature_dir}/{clip_id}_audio.pt"
            if os.path.exists(feature_path):
                continue
                
            # Process audio
            try:
                audio = self.process_audio(audio_path)
                torch.save(audio, feature_path)
                if idx % 100 == 0:
                    print(f"Processed audio {idx}/{len(self)}")
            except Exception as e:
                print(f"Error saving audio features for {clip_id}: {e}")
    
    def __getitem__(self, idx):

        clip_id = self.clips[idx]
        clip_data = self.data_df[self.data_df['clip_id'] == clip_id].iloc[0]
        audio_path = self.get_item_paths(clip_id)['audio']
        audio_features = self.get_item_paths(clip_id)['audio_features']

        if os.path.exists(audio_features):
            audio = torch.load(audio_features, weights_only=False)
        else:
            audio = self.process_audio(audio_path)
            
        label = self.process_label(clip_data)
        return {
            "clip_id": clip_id,
            "audio_inputs": audio,
            **label,
        }
        
class VideoDataset(BaseDataset):
    """Dataset for video-only processing using ViVIT"""
    
    def __init__(self, 
                 root: str, 
                 label_path: str, 
                 feature_dir: str = 'video_features',
                 precompute: bool = True,
                 **kwargs):
        
        # Initialize video processor before calling parent constructor
        self.video_processor = VivitImageProcessor.from_pretrained(
            "google/vivit-b-16x2-kinetics400"
        )
        
        # Call parent constructor
        super().__init__(root, label_path, feature_dir, precompute)
    
    def process_video(self, video_path: str) -> Dict[str, torch.Tensor]:
        return _process_video(video_path, self.video_processor)
    
    def _precompute_features(self):
        """Precompute video features for all clips"""
        print(f"Precomputing video features for {len(self)} clips...")
        for idx in range(len(self)):
            clip_id = self.clips[idx]
            video_path = self.get_item_paths(clip_id)['video']
            
            # Skip if already computed
            feature_path = f"{self.root}/{self.feature_dir}/{clip_id}_video.pt"
            if os.path.exists(feature_path):
                continue
                
            # Process video
            try:
                video = self.process_video(video_path)
                torch.save(video, feature_path)
                if idx % 100 == 0:
                    print(f"Processed video {idx}/{len(self)}")
            except Exception as e:
                print(f"Error saving video features for {clip_id}: {e}")
    
    def __getitem__(self, idx):
        clip_id = self.clips[idx]
        clip_data = self.data_df[self.data_df['clip_id'] == clip_id].iloc[0]
        video_path = self.get_item_paths(clip_id)['video']
        
        try:
            if self.feature_dir:
                # Load precomputed features
                video_file = f"{self.root}/{self.feature_dir}/{clip_id}_video.pt"
                if os.path.exists(video_file):
                    video = torch.load(video_file)
                else:
                    # Fallback to processing on-the-fly
                    video = self.process_video(video_path)
            else:
                # Process on-the-fly
                video = self.process_video(video_path)
                
            label = self.process_label(clip_data)
            return {
                "clip_id": clip_id,
                "video_inputs": video,
                **label,
            }
        except Exception as e:
            print(f"Error retrieving video item {clip_id}: {e}")
            raise ValueError(f"Error processing video {clip_id}: {e}")

class VideoAudioDataset(BaseDataset):
    """Multimodal dataset for both video and audio processing"""
    
    def __init__(self, 
                 root: str, 
                 label_path: str, 
                 feature_dir: str = 'multimodal_features',
                 precompute: bool = True,
                 sampling_rate: int = 16000,
                 **kwargs):
        
        # Initialize processors
        self.audio_processor = Wav2Vec2FeatureExtractor.from_pretrained(
            "facebook/wav2vec2-base-960h",
            sampling_rate=sampling_rate
        )
        
        self.video_processor = VivitImageProcessor.from_pretrained(
            "google/vivit-b-16x2-kinetics400"
        )
        
        self.target_sampling_rate = sampling_rate
        
        # Call parent constructor
        super().__init__(root, label_path, feature_dir, precompute)
    
    def process_audio(self, audio_path: str) -> Dict[str, torch.Tensor]:
        return _process_audio(audio_path, self.audio_processor)
    
    def process_video(self, video_path: str) -> Dict[str, torch.Tensor]:
        return _process_video(video_path, self.video_processor)
    
    def _precompute_features(self):
        """Precompute both audio and video features for all clips"""
        print(f"Precomputing multimodal features for {len(self)} clips...")
        for idx in range(len(self)):
            clip_id = self.clips[idx]
            paths = self.get_item_paths(clip_id)
            
            # Skip if both already computed
            audio_path = f"{self.root}/{self.feature_dir}/{clip_id}_audio.pt"
            video_path = f"{self.root}/{self.feature_dir}/{clip_id}_video.pt"
            
            if not os.path.exists(audio_path):
                try:
                    audio = self.process_audio(paths['audio'])
                    torch.save(audio, audio_path)
                except Exception as e:
                    print(f"Error saving audio features for {clip_id}: {e}")
            
            if not os.path.exists(video_path):
                try:
                    video = self.process_video(paths['video'])
                    torch.save(video, video_path)
                except Exception as e:
                    print(f"Error saving video features for {clip_id}: {e}")
                    
            if idx % 100 == 0:
                print(f"Processed {idx}/{len(self)} clips")
    
    def __getitem__(self, idx):
        clip_id = self.clips[idx]
        clip_data = self.data_df[self.data_df['clip_id'] == clip_id].iloc[0]
        paths = self.get_item_paths(clip_id)
        
        # Default to empty outputs in case of errors
        audio = {"input_values": torch.zeros(1, 16000)}
        video = {"pixel_values": torch.zeros(1, 16, 3, 224, 224)}
        error = None
        
        try:
            if self.feature_dir:
                # Load precomputed features
                audio_file = f"{self.root}/{self.feature_dir}/{clip_id}_audio.pt"
                video_file = f"{self.root}/{self.feature_dir}/{clip_id}_video.pt"
                
                if os.path.exists(audio_file):
                    audio = torch.load(audio_file)
                else:
                    # Fallback to processing on-the-fly
                    audio = self.process_audio(paths['audio'])
                    
                if os.path.exists(video_file):
                    video = torch.load(video_file)
                else:
                    # Fallback to processing on-the-fly
                    video = self.process_video(paths['video'])
            else:
                # Process on-the-fly
                audio = self.process_audio(paths['audio'])
                video = self.process_video(paths['video'])
        except Exception as e:
            error = str(e)
            print(f"Error retrieving multimodal item {clip_id}: {e}")
        
        label = self.process_label(clip_data)
        result = {
            "clip_id": clip_id,
            "audio_inputs": audio,
            "video_inputs": video,
            **label,
        }
        
        if error:
            result["error"] = error
            
        return result

def prep_dataset(config: DatasetConfig, split: str = "train", modality: Optional[str] = 'audio') -> Union[AudioDataset, VideoDataset, VideoAudioDataset]:

    if split == "test":
        config.use_gold = True
    if modality == "audio":
        return AudioDataset(**vars(config), split=split)
    elif modality == "video":
        return VideoDataset(**vars(config), split=split)
    elif modality == "multimodal":
        return VideoAudioDataset(**vars(config), split=split)
    else:
        raise ValueError(f"Unsupported modality: {modality}")


if __name__ == "__main__":

    config = DatasetConfig(
        root="data/clips",
        label_path="data/clips/labels/A1_labels.csv",
        feature_dir="features",
        precompute=False,
        sampling_rate=16000
    )
    
    dataset = prep_dataset(config, split="train", modality="multimodal")
    print(f"Loaded {len(dataset)} audio clips for training")
    
    for i in range(5):
        item = dataset[i]
        # pprint.pprint(item)/
        print(item['video_inputs']['pixel_values'].shape)