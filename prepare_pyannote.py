import pandas as pd
import numpy as np
import yaml
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import soundfile as sf
from pydub import AudioSegment
import wave
import argparse
from tqdm import tqdm

STUTTER_COLUMNS = ['SR', 'ISR', 'MUR', 'P', 'B', 'V', 'FG', 'HM', 'ME']
STUTTER_MAPPING = {
            'SR': 'SoundRepetition',
            'ISR': 'IncompeleteSyllableRepetition',
            'MUR': 'MultisyllabicUnitRepetition',
            'P': 'Prolongation',
            'B': 'Block',
            'NV': 'NonVerbal',
            'V': 'Verbal',
            'FG': 'FacialGrimaces',
            'HM': 'HeadMovement',
            'ME': 'MovementOfExtremities',
            }

# Stuttering types mapping
class HierarchyDefinition:
    """Defines the hierarchical structure of stuttering types"""
    def __init__(self):
        self.hierarchy = {
            'stutter': {
                'repetition': {
                    'children': [
                        'SoundRepetition',
                        'IncompeleteSyllableRepetition',
                        'MultisyllabicUnitRepetition'
                    ]
                },
                'flow_disruption': {
                    'children': [
                        'Prolongation',
                        'Block'
                    ]
                },
                'secondary': {
                    'children': [
                        'NonVerbal',
                        'Verbal',    
                        'FacialGrimaces',
                        'HeadMovement',
                        'MovementOfExtremities'
                    ]
                }
            }
        }
        
        # Build flat list of all categories
        self.all_categories = self._build_category_list()
        
        # Build parent mapping
        self.parent_mapping = self._build_parent_mapping()
    
    def _build_category_list(self) -> List[str]:
        """Build flat list of all categories including parents"""
        categories = ['stutter']
        
        for level1, level1_data in self.hierarchy['stutter'].items():
            categories.append(level1)
            categories.extend(level1_data['children'])
            
        return categories
    
    def _build_parent_mapping(self) -> Dict[str, List[str]]:
        """Build mapping from child to list of parents"""
        mapping = {}
        
        # Add level 1 parents
        for level1, level1_data in self.hierarchy['stutter'].items():
            mapping[level1] = ['stutter']
            
            # Add level 2 parents
            for child in level1_data['children']:
                mapping[child] = ['stutter', level1]
                
        return mapping
    
    def get_parent_labels(self, child_label: str) -> List[str]:
        """Get all parent labels for a given child label"""
        return self.parent_mapping.get(child_label, [])
    
@dataclass
class StutterEvent:
    """Data class for storing stutter event information with both binary and subtype labels"""
    start: float
    end: float
    types: List[str]  # Will include 'stutter' and specific types
    annotator: str
    media_id: Optional[str] = None

@dataclass
class DatasetConfig:
    """Configuration for dataset setup"""
    name: str
    input_dir: Path
    database_dir: Path
    exclusions_csv: Optional[Path] = None
    annotators: List[str] = None
    add_parents: bool = True

    def __post_init__(self):
        if self.annotators is None:
            self.annotators = ['A1', 'A2', 'A3', 'aggregated']
        
        # Set up paths
        self.audio_dir = self.input_dir / f"{self.name}/audio"
        self.video_dir = self.input_dir / f"{self.name}/video"
        self.rttm_dir = self.database_dir / f"{self.name}/rttm"
        self.uem_dir = self.database_dir / f"{self.name}/uem"
        self.lst_dir = self.database_dir / f"{self.name}/lst"
        
        # Input files
        # set the exclusions_csv to the interview folder if it exists else set it to none
        self.exclusions_csv = self.input_dir / f"{self.name}/exclusions.csv" if (self.input_dir / f"{self.name}/exclusions.csv").exists() else None
        self.annotations_csv = self.input_dir / f"{self.name}/total_dataset_final.csv"
        self.split_json = self.input_dir / f"{self.name}/split.json"

class AnnotationProcessor:
    """Process stuttering annotations including both binary and subtype labels"""
    
    def __init__(self, csv_path: Path, add_parents: bool = True):
        self.df = pd.read_csv(csv_path)
        # Convert milliseconds to seconds
        self.df['start'] = self.df['start'] / 1000
        self.df['end'] = self.df['end'] / 1000
        self.annotators = self.df['annotator'].unique()
        self.hierarchy = HierarchyDefinition()
        self.add_parents = add_parents
        
    def process_events(self, annotator: str) -> List[StutterEvent]:
        """Extract stuttering events with hierarchical labels"""
        annotator_df = self.df[self.df['annotator'] == annotator]
        events = []
        
        for _, row in annotator_df.iterrows():
            event_types = set()
            
            # Process leaf node labels
            for col in STUTTER_COLUMNS:
                if row[col] == 1:
                    leaf_type = STUTTER_MAPPING[col]
                    event_types.add(leaf_type)
                    # Add all parent labels
                    if self.add_parents:
                        event_types.update(self.hierarchy.get_parent_labels(leaf_type))
            
            if event_types:
                event = StutterEvent(
                    start=row['start'],
                    end=row['end'],
                    types=list(event_types),
                    annotator=annotator,
                    media_id=row['media_file']
                )
                events.append(event)
                
        return events
    
    def process_aggregated_events(self, annotators: List[str], 
                                agreement_threshold: float = 0.5) -> List[StutterEvent]:
        """Create aggregated events with hierarchical labels"""
        annotator_df = self.df[self.df['annotator'].isin(annotators)]
        item_groups = annotator_df.groupby(['item', 'media_file'])
        events = []
        
        for (item, media_id), group in item_groups:
            type_counts = defaultdict(int)
            
            # Count annotations for all types including parents
            for annotator in annotators:
                annotator_data = group[group['annotator'] == annotator]
                if not annotator_data.empty:
                    for col in STUTTER_COLUMNS:
                        if annotator_data[col].iloc[0] == 1:
                            leaf_type = STUTTER_MAPPING[col]
                            type_counts[leaf_type] += 1
                            # Count parent labels too
                            if self.add_parents:
                                for parent in self.hierarchy.get_parent_labels(leaf_type):
                                    type_counts[parent] += 1
            
            # Calculate agreements
            n_annotators = len(annotators)
            min_agreeing = int(n_annotators * agreement_threshold)
            
            # Get all types that meet agreement threshold
            agreed_types = [
                stype for stype, count in type_counts.items()
                if count >= min_agreeing
            ]
            
            if agreed_types:
                start = group['start'].mean()
                end = group['end'].mean()
                
                event = StutterEvent(
                    start=start,
                    end=end,
                    types=agreed_types,
                    annotator='aggregated',
                    media_id=media_id
                )
                events.append(event)
        
        return events

class RTTMWriter:
    """Handles creation of RTTM files with combined binary and subtype labels"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def create_rttm(self, media_id: str, events: List[StutterEvent], annotator: str) -> Path:
        """Create RTTM file containing both binary and subtype labels"""
        output_file = self.output_dir / f"{media_id}_{annotator}.rttm"
            
        with open(output_file, 'w') as f:
            for event in events:
                if event.media_id == media_id:
                    duration = event.end - event.start
                    # Write all types (including binary 'stutter' label) on separate lines
                    for stype in event.types:
                        line = f"SPEAKER {media_id} 1 {event.start:.3f} {duration:.3f} <NA> <NA> {stype} <NA> <NA>\n"
                        f.write(line)
                    
        return output_file

class UEMWriter:
    """Handles creation of UEM files with exclusions support"""
    
    def __init__(self, 
                 output_dir: Path, 
                 audio_dir: Path,
                 exclusions_csv: Optional[Path] = None):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir = audio_dir
        self.exclusions = self._load_exclusions(exclusions_csv) if exclusions_csv else {}
        
    def _load_exclusions(self, exclusions_csv: Path) -> Dict[str, List[Tuple[float, float]]]:
        """Load segments to exclude from CSV file"""
        exclusions = defaultdict(list)
        df = pd.read_csv(exclusions_csv)
        
        required_cols = ['media_file', 'start', 'end']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Exclusions CSV must contain columns: {required_cols}")
        
        for _, row in df.iterrows():
            exclusions[row['media_file']].append((float(row['start']), float(row['end'])))
            
        for media_file in exclusions:
            exclusions[media_file].sort()
            
        return dict(exclusions)
    
    def _get_audio_length(self, media_id: str) -> float:
        """Get length of audio file in seconds"""
        extensions = ['.wav', '.mp3', '.flac']
        
        for ext in extensions:
            audio_path = self.audio_dir / f"{media_id}{ext}"
            if audio_path.exists():
                try:
                    if ext == '.wav':
                        with wave.open(str(audio_path), 'rb') as wav_file:
                            frames = wav_file.getnframes()
                            rate = wav_file.getframerate()
                            duration = frames / float(rate)
                            return duration
                    elif ext == '.mp3':
                        audio = AudioSegment.from_mp3(str(audio_path))
                        return len(audio) / 1000.0
                    elif ext == '.flac':
                        info = sf.info(str(audio_path))
                        return info.duration
                except Exception as e:
                    print(f"Error reading {audio_path}: {e}")
                    continue
                    
        raise FileNotFoundError(f"No audio file found for {media_id}")
    
    def _merge_intervals(self, intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Merge overlapping intervals"""
        if not intervals:
            return []
            
        merged = []
        intervals.sort()
        current_start, current_end = intervals[0]
        
        for start, end in intervals[1:]:
            if start <= current_end:
                current_end = max(current_end, end)
            else:
                merged.append((current_start, current_end))
                current_start, current_end = start, end
                
        merged.append((current_start, current_end))
        return merged
    
    def create_uem(self, media_id: str, annotator: str) -> Path:
        """Create UEM file with evaluable segments"""
        output_file = self.output_dir / f"{media_id}.uem"
        
        # Get audio duration
        audio_length = self._get_audio_length(media_id)
        
        # Get evaluable segments
        if media_id not in self.exclusions:
            segments = [(0.0, audio_length)]
        else:
            segments = [(0.0, audio_length)]
            for excl_start, excl_end in self.exclusions[media_id]:
                new_segments = []
                for seg_start, seg_end in segments:
                    if excl_end <= seg_start or excl_start >= seg_end:
                        new_segments.append((seg_start, seg_end))
                    else:
                        if seg_start < excl_start:
                            new_segments.append((seg_start, excl_start))
                        if seg_end > excl_end:
                            new_segments.append((excl_end, seg_end))
                segments = new_segments
            
            segments = self._merge_intervals(segments)
        
        with open(output_file, 'w') as f:
            for start, end in segments:
                line = f"{media_id} 1 {start:.3f} {end:.3f}\n"
                f.write(line)
                
        return output_file

class LSTFileCreator:
    """Creates LST files for dataset splits"""
    
    def __init__(self, split_json: Path, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.splits = self._load_splits(split_json)
        
    def _load_splits(self, split_json: Path) -> Dict[str, List[str]]:
        """Load and validate split configuration"""
        with open(split_json, 'r') as f:
            splits = json.load(f)
            
        required_splits = {'train', 'val', 'test'}
        if not all(split in splits for split in required_splits):
            raise ValueError(f"Split JSON must contain all of: {required_splits}")
            
        all_files = []
        for files in splits.values():
            all_files.extend(files)
        duplicates = [x for x in all_files if all_files.count(x) > 1]
        if duplicates:
            raise ValueError(f"Found duplicate media files across splits: {set(duplicates)}")
            
        return splits
    
    def create_lst_files(self):
        """Create LST files for each split"""
        for split_name, media_files in self.splits.items():
            media_files.sort()
            output_file = self.output_dir / f"{split_name}.lst"
            with open(output_file, 'w') as f:
                f.write('\n'.join(media_files))
            print(f"Created {output_file} with {len(media_files)} entries")

class DatasetSetupManager:
    """Manages the setup of both binary and multilabel datasets"""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.heirarchy = HierarchyDefinition()
        
    def setup(self):
        """Set up the complete dataset structure"""
        self._verify_inputs()
        self._create_directory_structure()
        self._setup_dataset()
        self._update_database_yaml()
        
    def _verify_inputs(self):
        """Verify all required input files exist"""
        # if not (self.config.input_dir / "audio").exists():
        #     raise FileNotFoundError(f"Input audio directory not found: {self.config.input_dir / 'audio'}")
        # if not (self.config.input_dir / "video").exists():
        #     raise FileNotFoundError(f"Input video directory not found: {self.config.input_dir / 'video'}")
        if not self.config.annotations_csv.exists():
            raise FileNotFoundError(f"Annotations CSV not found: {self.config.annotations_csv}")
        if not self.config.split_json.exists():
            raise FileNotFoundError(f"Split JSON not found: {self.config.split_json}")
            
    def _create_directory_structure(self):
        import os
        """Create the directory structure and symlinks"""
        # Create base directories
        self.config.database_dir.mkdir(parents=True, exist_ok=True)
        
        self.config.rttm_dir.mkdir(parents=True, exist_ok=True)
        self.config.uem_dir.mkdir(parents=True, exist_ok=True)
        self.config.lst_dir.mkdir(parents=True, exist_ok=True)

    def _setup_dataset(self):
        """Process annotations and create dataset files"""
        # Initialize processors
        processor = AnnotationProcessor(self.config.annotations_csv, add_parents=self.config.add_parents)
        rttm_writer = RTTMWriter(self.config.rttm_dir)
        uem_writer = UEMWriter(self.config.uem_dir, self.config.audio_dir, self.config.exclusions_csv)
        lst_creator = LSTFileCreator(self.config.split_json, self.config.lst_dir)
        
        # Create LST files
        print(f"\nCreating LST files for dataset...")
        lst_creator.create_lst_files()
        
        # Load split information
        with open(self.config.split_json, 'r') as f:
            splits = json.load(f)
        all_media_files = []
        for split_files in splits.values():
            all_media_files.extend(split_files)
            
        # Process individual annotators
        for media_id in tqdm(all_media_files, desc="Processing annotations", total=len(all_media_files)):
            for annotator in self.config.annotators:
                if annotator != 'aggregated':
                    events = processor.process_events(annotator)
                    if events:
                        # print(f"Creating RTTM for {media_id}_{annotator}")
                        rttm_writer.create_rttm(media_id, events, annotator)

            uem_writer.create_uem(media_id, annotator)
        
        # Process aggregated annotations
        if 'aggregated' in self.config.annotators:
            target_annotators = [ann for ann in self.config.annotators if ann not in ['aggregated', 'bau', 'mas', 'sad', 'Gold']]
            for media_id in tqdm(all_media_files, desc="Processing aggregated annotations", total=len(all_media_files)):
                agg_events = processor.process_aggregated_events(target_annotators)
                if agg_events:
                    rttm_writer.create_rttm(media_id, agg_events, 'aggregated')
                uem_writer.create_uem(media_id, 'aggregated')
            
    def _update_database_yaml(self):
        """Create or update the database.yml file"""
        database_yml = self.config.database_dir / f"database.yml"
        
        # Load existing configuration if it exists
        if database_yml.exists():
            with open(database_yml, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = {
                'Databases': {},
                'Protocols': {}
            }
        
        # Update databases section
        config['Databases'][f'{self.config.name.capitalize()}'] = str(self.config.audio_dir.absolute() / "{uri}.wav")
        
        # Update protocols section
        if f'{self.config.name.capitalize()}' not in config['Protocols']:
            config['Protocols'][f'{self.config.name.capitalize()}'] = {
                'SpeakerDiarization': {}
            }
        
        for annotator in self.config.annotators:
            config['Protocols'][f'{self.config.name.capitalize()}']['SpeakerDiarization'][annotator] = {
                'classes': list(self.heirarchy.all_categories) if self.config.add_parents else list(STUTTER_COLUMNS),
                'scope': 'file',
                'train': {
                    'uri': str(self.config.lst_dir / 'train.lst'),
                    'annotation': str(self.config.rttm_dir / f'{{uri}}_{annotator}.rttm'),
                    'annotated': str(self.config.uem_dir / f'{{uri}}.uem')
                },
                'development': {
                    'uri': str(self.config.lst_dir / 'val.lst'),
                    'annotation': str(self.config.rttm_dir / f'{{uri}}_{annotator}.rttm'),
                    'annotated': str(self.config.uem_dir / f'{{uri}}.uem')
                },
                'test': {
                    'uri': str(self.config.lst_dir / 'test.lst'),
                    'annotation': str(self.config.rttm_dir / f'{{uri}}_Gold.rttm'),
                    'annotated': str(self.config.uem_dir / f'{{uri}}.uem')
                }
            }
        
        # Write updated configuration back to file
        with open(database_yml, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
            
def setup_dataset(args):
    """Set up the dataset with combined binary and multilabel annotations"""
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    # Configuration
    names = args.name.split('+')
    for name in names:
        config = DatasetConfig(
            name = name,
            input_dir=input_dir,
            database_dir=output_dir,
            annotators=['A1', 'A2', 'A3', 'bau','sad', 'mad', 'Gold'],
            add_parents=True
        )
    
        # Set up dataset
        manager = DatasetSetupManager(config)
        print("\nSetting up dataset...")
        manager.setup()
    
    print("\nDataset has been set up successfully!")
    print(f"Dataset directory structure:\n")
    print(f"  {output_dir}/")
    print("    ├── database.yml")
    print("    ├── rttm/")
    print("    ├── uem/")
    print("    └── lst/")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Prepare stuttering dataset")
    parser.add_argument("--name", type=str, default="interview+reading", help="Name of the dataset")
    parser.add_argument("--input_dir", type=str, default="../StED/data", help="Directory containing input files")
    parser.add_argument("--output_dir", type=str, default="./output/pyannote", help="Directory to save dataset")

    args = parser.parse_args()
    
    # Set up dataset
    setup_dataset(args)
    