import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Union
from configs import TrainingConfig, DatasetConfig
import h5py
from sklearn.utils import resample

class BaseDataset(Dataset):
    """Base dataset class with common functionality for all modalities"""
    
    labels = "SR,ISR,MUR,P,B,V,FG,HM,ME".split(",")
    primary_labels = "SR,ISR,MUR,P,B".split(",")
    secondary_labels = "V,FG,HM,ME".split(",")
    
    def __init__(self, 
                 root: str, 
                 annotator: str,
                 split: str = "train",
                 label: Optional[List[str]] = None):
        """
        Initialize base dataset
        
        Args:
            root: Root directory containing data
            label_path: Path to CSV file with labels
        """
        self.root = root
        self.annotator = annotator
        self.label = label 
        self.split = split

        self.data_df = self.prep_df()
        self.clips = self.data_df['clip_id']
         
    def prep_df(self):
        label_path = os.path.join(self.root, "labels", f"{self.annotator}.csv")
        df = pd.read_csv(label_path)
        df = df[df['split'] == self.split].reset_index(drop=True)

        if self.split == "test":
            return df
        
        if self.label:
    
            df_majority = df[df[self.label]==0]
            df_minority = df[df[self.label]==1]

            df_minority_upsampled = resample(df_minority,
                                                replace=True,    
                                                n_samples=len(df_majority),    
                                                random_state=42)
            df_upsampled = pd.concat([df_majority, df_minority_upsampled])
            df_upsampled = df_upsampled.sample(frac=1, random_state=42).reset_index(drop=True)

            return df_upsampled
        else:
            return df

    def read_h5(self, h5_path: str, clip_id: str) -> Dict[str, torch.Tensor]:
        """Load all features for a clip_id from an HDF5 file."""
        features = {}
        with h5py.File(h5_path, 'r') as h5f:
            if clip_id not in h5f:
                raise KeyError(f"Clip {clip_id} not found in {h5_path}")
            group = h5f[clip_id]
            for k in group.keys():
                arr = group[k][()]
                features[k] = torch.tensor(arr, dtype=torch.float32)  
        
        return features
    
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
    
    def __len__(self) -> int:
        return len(self.clips)
    
    def get_item_paths(self, clip_id):
        """Get paths for a specific clip ID"""
        return {
            'audio': f"{self.root}/audios/{clip_id}.h5",
            'video': f"{self.root}/videos/{clip_id}.h5",
        }

class AudioDataset(BaseDataset):
    """Dataset for audio-only processing using Wav2Vec2"""
    
    def __init__(self, 
                 root: str, 
                 annotator: str, 
                 sampling_rate: int = 16000,
                 split: str = "train",
                 label: Optional[List[str]] = None,
                 **kwargs):
        
        self.target_sampling_rate = sampling_rate
        
        super().__init__(root, annotator, split, label)
    
    def __getitem__(self, idx):

        clip_id = self.clips[idx]
        clip_data = self.data_df[self.data_df['clip_id'] == clip_id].iloc[0]
        media_file = clip_data['media_file']
        task = clip_data['task']
        audio_path = self.get_item_paths(f"{task}_{media_file}")['audio']

        if os.path.exists(audio_path):
            audio = self.read_h5(audio_path, clip_id)
        else:
            print(f"Audio file not found for clip {clip_id}")
            audio = torch.zeros(1, self.target_sampling_rate)  # Default to empty audio

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
                 annotator: str,
                 split: str = "train",
                 **kwargs):
        
        super().__init__(root, annotator=annotator, split=split, label=None)

    def __getitem__(self, idx):
        clip_id = self.clips[idx]
        clip_data = self.data_df[self.data_df['clip_id'] == clip_id].iloc[0]
        media_file = clip_data['media_file']
        task = clip_data['task']
        video_path = self.get_item_paths(f"{task}_{media_file}")['video']

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found for clip {clip_id}: {video_path}")
        
        video = self.read_h5(video_path, clip_id)
        label = self.process_label(clip_data)

        return {
            "clip_id": clip_id,
            "video_inputs": video,
            **label,
        }
    
class VideoAudioDataset(BaseDataset):
    """Multimodal dataset for both video and audio processing"""
    
    def __init__(self, 
                 root: str, 
                 annotator: str, 
                 split: str = "train",
                 sampling_rate: int = 16000,
                 **kwargs):
        
        self.target_sampling_rate = sampling_rate
        super().__init__(root, annotator=annotator, split=split, label=None)

    def __getitem__(self, idx):
        clip_id = self.clips[idx]
        
        clip_data = self.data_df[self.data_df['clip_id'] == clip_id].iloc[0]
        media_file = clip_data['media_file']
        task = clip_data['task']
        paths = self.get_item_paths(f"{task}_{media_file}")
        audio_path = paths['audio']
        video_path = paths['video']

        if not os.path.exists(audio_path) or not os.path.exists(video_path):
            raise FileNotFoundError(f"file not found for clip {clip_id}: {audio_path}, {video_path}")
        
        audio_inputs = self.read_h5(audio_path, clip_id) 
        video_inputs = self.read_h5(video_path, clip_id)
        label = self.process_label(clip_data)
        breakpoint()
        return {
            "clip_id": clip_id,
            "audio_inputs": audio_inputs,
            "video_inputs": video_inputs,
            **label,
        }
    

def prep_dataset(config: DatasetConfig, split: str = "train", modality: Optional[str] = 'audio') -> Union[AudioDataset, VideoDataset, VideoAudioDataset]:

    if modality == "audio":
        return AudioDataset(**vars(config), split=split)
    elif modality == "video":
        return VideoDataset(**vars(config), split=split)
    elif modality == "multimodal":
        return VideoAudioDataset(**vars(config), split=split)
    else:
        raise ValueError(f"Unsupported modality: {modality}")

def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    
    # collator to stack tensors
    collated = {}
    for key in batch[0].keys():
        if isinstance(batch[0][key], torch.Tensor):
            collated[key] = torch.stack([item[key] for item in batch])
        elif isinstance(batch[0][key], dict):
            collated[key] = collate_fn([item[key] for item in batch])
        else:
            collated[key] = [item[key] for item in batch]

    return collated

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    config = DatasetConfig(
        root="data/clips",
        annotator="A1",
        sampling_rate=16000
    )
    
    dataset = prep_dataset(config, split="train", modality="multimodal")
    
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=collate_fn)
    for batch in data_loader:
        print(batch['audio_inputs']['input_values'].shape, batch['video_inputs']['pixel_values'].shape)
        break  # Just print the first batch for testing