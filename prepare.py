import os
import pandas as pd
import numpy as np
from pydub import AudioSegment
import av  
import hashlib  
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm  
import torch
import torchvision
from typing import Dict
from transformers import Wav2Vec2Processor, VivitImageProcessor
import h5py


video_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")
audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

def agg_fn():
    return{
        'media_file': 'first',
        'task': 'first',
        'split': 'first',
        'annotator': 'first',
        'start_time': 'first',
        'end_time': 'first',
        'annotation_start': 'min',
        'annotation_end': 'max',
        'SR': 'max',
        'ISR': 'max',
        'MUR': 'max',
        'P': 'max',
        'B': 'max',
        'V': 'max',
        'FG': 'max',
        'HM': 'max',
        'ME': 'max',
        'T': 'max'
    }

def load_exclusions(exclusions_path):

    exclusions = {}
    if os.path.exists(exclusions_path):
        df = pd.read_csv(exclusions_path)
        for _, row in df.iterrows():
            if row['media_file'] not in exclusions:
                exclusions[row['media_file']] = []
            exclusions[row['media_file']].append((row['start'], row['end']))
    return exclusions

def is_segment_excluded(start_time, end_time, excluded_segments):

    if excluded_segments is None:
        return False
        
    for ex_start, ex_end in excluded_segments:
        # Check for overlap
        if not (end_time <= ex_start or start_time >= ex_end):
            return True
    return False

def _process_audio(audio: str, processor) -> Dict[str, torch.Tensor]:
    """Process audio file using Wav2Vec2 processor"""
    
    # Convert to mono if stereo
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
        
    # Process with Wav2Vec2 processor
    processed = processor(
        audio.numpy(),  
        sampling_rate=16000,  # Ensure consistent sampling rate
        return_tensors="pt",
        
    )
    
    return processed

def deterministic_clip_id(task, media_file, start_time, end_time, clip_duration):
    """
    Generate a deterministic, reusable clip ID based on media_file, start_time, end_time, and clip_duration.
    """
    id_str = f"{task}_{media_file}_{start_time:.3f}_{end_time:.3f}"
    # return hashlib.md5(id_str.encode('utf-8')).hexdigest()
    return id_str

def process_video(args):
    """
    Process all clips for a single video file in one thread, extracting audio and video features.
    Uses PyAV for video processing and saves preprocessed features directly without saving MP4 files.
    """
    media_file, task, group, excluded_segments, video_path, output_dir, clip_duration, overlap, segment_to_clip_map = args
    
    successful_clips = set()
    video_output_dir = os.path.join(output_dir, 'videos')
    audio_output_dir = os.path.join(output_dir, 'audios')
    os.makedirs(video_output_dir, exist_ok=True)
    os.makedirs(audio_output_dir, exist_ok=True)

    # Prepare HDF5 files for this video
    audio_h5_path = os.path.join(audio_output_dir, f'{task}_{media_file}.h5')
    video_h5_path = os.path.join(video_output_dir, f'{task}_{media_file}.h5')
    audio_h5 = h5py.File(audio_h5_path, 'a')
    video_h5 = h5py.File(video_h5_path, 'a')

    # Process video and audio with PyAV and save preprocessed features
    try:
        with av.open(video_path) as container:
            
            video_stream = next(s for s in container.streams if s.type == 'video')
            audio_stream = next((s for s in container.streams if s.type == 'audio'), None)
            
            fps = float(video_stream.average_rate)
            duration = container.duration / av.time_base

            # Extract all audio frames first
            audio_frames = []
            if audio_stream:
                container.seek(0)
                for frame in container.decode(audio=0):
                    audio_frames.append(frame.to_ndarray())
            
            # Concatenate audio frames and convert to tensor
            if audio_frames:
                audio_array = np.concatenate(audio_frames, axis=1)
                audio_tensor = torch.from_numpy(audio_array).float()
                
                # Convert stereo to mono if needed
                if audio_tensor.shape[0] > 1:
                    audio_tensor = torch.mean(audio_tensor, dim=0, keepdim=True)
                
                # Resample to 16kHz if needed
                original_sr = audio_stream.sample_rate
                if original_sr != 16000:
                    import torchaudio
                    resampler = torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=16000)
                    audio_tensor = resampler(audio_tensor)
            else:
                # No audio - create silence
                audio_tensor = torch.zeros(1, int(duration * 16000))

            stride = clip_duration - overlap
            start_times = np.arange(0, duration - clip_duration, stride)
            
            video_clips = []
            for start_time in start_times:
                end_time = start_time + clip_duration
                if is_segment_excluded(start_time, end_time, excluded_segments):
                    continue
                # Use deterministic clip ID
                clip_id = deterministic_clip_id(task, media_file, start_time, end_time, clip_duration)
                segment_key = (task, media_file, start_time, end_time)
                segment_to_clip_map[segment_key] = clip_id
                video_clips.append((clip_id, start_time, end_time))

            print(f"Found {len(video_clips)} clips in {video_path}")
            for clip_id, start_time, end_time in video_clips:

                # Skip if already processed
                if clip_id in audio_h5 or clip_id in video_h5:
                    successful_clips.add(clip_id)
                    print(f"Skipping already processed clip {clip_id}")
                    continue

                # Process and save audio features
                start_sample = int(start_time * 16000)
                end_sample = int(end_time * 16000)
                audio_clip = audio_tensor[:, start_sample:end_sample]
              
                audio_features = _process_audio(audio_clip, audio_processor)
                
                # Save as float16
                for k, v in audio_features.items():
                    arr = v.cpu().numpy().astype('float16')
                    ds_name = f"{clip_id}/{k}"
                    if ds_name in audio_h5:
                        del audio_h5[ds_name]
                    audio_h5.create_dataset(ds_name, data=arr, compression='gzip')

                # Extract frames directly for video features
                start_frame = int(start_time * fps)
                end_frame = int(end_time * fps)
                frames_to_extract = end_frame - start_frame
                # Extract frames needed for video features
                container.seek(int(start_time * 1000000), any_frame=False, stream=video_stream)
                
                frames = []
                for i, frame in enumerate(container.decode(video=0)):
                    if i >= frames_to_extract:
                        break
                    # Convert PyAV frame to numpy array
                    frame_array = frame.to_ndarray(format="rgb24")
                    frames.append(frame_array)
                
                if len(frames) < 32:  # Pad if we don't have enough frames
                    last_frame = frames[-1] if frames else np.zeros((video_stream.height, video_stream.width, 3), dtype=np.uint8)
                    frames.extend([last_frame] * (32 - len(frames)))
                
                # Sample or trim to exactly 32 frames
                if len(frames) != 32:
                    indices = np.linspace(0, len(frames) - 1, 32).astype(int)
                    frames = [frames[i] for i in indices]
                
                # Convert frames to PIL images for the processor
                pil_frames = [torchvision.transforms.ToPILImage()(torch.from_numpy(frame).permute(2,0,1)) for frame in frames]
                
                video_features = video_processor(images=pil_frames, return_tensors="pt")
                for k, v in video_features.items():
                    arr = v.cpu().numpy().astype('float16')
                    ds_name = f"{clip_id}/{k}"
                    if ds_name in video_h5:
                        del video_h5[ds_name]
                    video_h5.create_dataset(ds_name, data=arr, compression='gzip')
        
                successful_clips.add(clip_id)
                
    except Exception as e:
        print(f"Error processing video {video_path}: {str(e)}")
    finally:
        # Close HDF5 files
        audio_h5.close()
        video_h5.close()

    return video_clips, successful_clips

def create_overlapping_clips(input_df, base_path, output_dir='clips', clip_duration=5, overlap=2, max_workers=4):
    """
    Create overlapping clips using one thread per video file.
    
    Args:
        input_df: Pandas DataFrame containing annotation data
        base_path: Base path where media files are stored
        output_dir: Directory to store output clips and labels
        clip_duration: Duration of each clip in seconds
        overlap: Overlap duration between clips in seconds
        max_workers: Maximum number of threads to use
    """
    
    # Create output directories
    os.makedirs(os.path.join(output_dir, 'audios'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'videos'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)

    # Load exclusions
    exclusions = load_exclusions('data/fluencybank_aws/interview/exclusions.csv')
    grouped = input_df.groupby(['media_file','task'])

    clips_metadata = []
    segment_to_clip_map = {}

    # Prepare arguments for each video file
    video_args = []
    for (media_file, task), group in tqdm(grouped, desc="Preparing video tasks"):
        excluded_segments = exclusions.get(media_file) if task == 'interview' else None
        video_path = os.path.join(base_path, f'{task}', 'videos', f'{media_file}.mp4')
        if not os.path.exists(video_path):
            continue
        video_args.append((media_file, task, group, excluded_segments, video_path, output_dir, clip_duration, overlap, segment_to_clip_map))

    print(f"\nFound {len(video_args)} video files to process")
    # Process each video file in parallel
    all_video_clips = []
    all_successful_clips = set()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_video, args) for args in video_args]
        with tqdm(total=len(futures), desc="Processing videos") as pbar:
            for future in as_completed(futures):
                video_clips, successful_clips = future.result()
                all_video_clips.extend(video_clips)
                all_successful_clips.update(successful_clips)
                pbar.update(1)
    

    # Create labels for successful clips
    meta_cols = ['media_file', 'task', 'split', 'start_time', 'end_time']
    label_cols = ['SR', 'ISR', 'MUR', 'P', 'B', 'V', 'FG', 'HM', 'ME', 'T']
    clips_metadata = []
    all_annotators = input_df['annotator'].unique()
    
    for segment_key, clip_id in tqdm(segment_to_clip_map.items(), desc="Creating labels"):
        seg_task, seg_media_file, start_time, end_time = segment_key
        
        # Get the group for this segment
        group = grouped.get_group((seg_media_file, seg_task)) if (seg_media_file, seg_task) in grouped.groups else pd.DataFrame()
        if group.empty:
            continue
            
        split = group['split'].iloc[0]
        
        # Find overlapping annotations
        clip_labels = group[
            ((group['start'] >= start_time) & (group['start'] < end_time)) |
            ((group['end'] > start_time) & (group['end'] <= end_time)) |
            ((group['start'] <= start_time) & (group['end'] >= end_time))
        ]
        
        # For test split, only use Gold annotator
        if split == 'test':
            annotators_to_process = ['Gold']
        else:
            annotators_to_process = all_annotators
        
        # Create entries for each required annotator
        for annotator in annotators_to_process:
            annotator_labels = clip_labels[clip_labels['annotator'] == annotator]
            
            if annotator_labels.empty:
                # No annotations for this annotator - create zero entry
                clips_metadata.append({
                    'clip_id': clip_id,
                    'media_file': seg_media_file,
                    'task': seg_task,
                    'split': split,
                    'annotator': annotator,
                    'annotation_start': np.nan,
                    'annotation_end': np.nan,
                    'start_time': start_time,
                    'end_time': end_time,
                    **{col: 0 for col in label_cols}
                })
            else:
                # Add entries for each annotation from this annotator
                for _, row in annotator_labels.iterrows():
                    clips_metadata.append({
                        'clip_id': clip_id,
                        'media_file': seg_media_file,
                        'task': seg_task,
                        'split': split,
                        'annotator': annotator,
                        'annotation_start': row['start'],
                        'annotation_end': row['end'],
                        'start_time': start_time,
                        'end_time': end_time,
                        **{col: row[col] for col in label_cols}
                    })

    # Create and save labels DataFrame
    labels_df = pd.DataFrame(clips_metadata)
    
    # Separate test and non-test data
    test_df = labels_df[(labels_df['annotator'] == 'Gold') & (labels_df['split'] == 'test')]
    test_df = test_df.groupby('clip_id').agg(agg_fn()).reset_index()
    
    # Process each non-Gold annotator
    annotators = [a for a in all_annotators if a != 'Gold']
    all_annotator_dfs = []
    
    for annotator in annotators:
        # Get non-test data for this annotator (including Gold for reference)
        annotator_df = labels_df[
            (labels_df['split'] != 'test') & 
            (labels_df['annotator'].isin(['Gold', annotator]))
        ]
        annotator_df = annotator_df.groupby('clip_id').agg(agg_fn()).reset_index()
        
        # Save combined data (annotator + test)
        combined_df = pd.concat([annotator_df, test_df]).sort_values(["media_file", "start_time"])
        combined_df.to_csv(os.path.join(output_dir, 'labels', f'{annotator}.csv'), index=False)
        
        all_annotator_dfs.append(annotator_df)
    
    # Create majority vote labels
    total_annotator_df = pd.concat(all_annotator_dfs, ignore_index=True)
    
    def majority_vote(group):
        result = group.iloc[0].copy()  # Start with first row for metadata
        result['annotator'] = 'MAJ'
        
        # Calculate majority vote for each label column
        for col in label_cols:
            vals = group[col].dropna().astype(int)
            if len(vals) > 0:
                result[col] = vals.mode().iloc[0] if not vals.mode().empty else 0
            else:
                result[col] = 0
        return result
    
    maj_df = total_annotator_df.groupby('clip_id').apply(majority_vote).reset_index(drop=True)
    
    # Save majority vote + test data
    maj_combined = pd.concat([maj_df, test_df]).sort_values(["media_file", "start_time"])
    maj_combined.to_csv(os.path.join(output_dir, 'labels', 'MAJ.csv'), index=False)

    return labels_df

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Create overlapping clips from dataset")
    # parser.add_argument('--input_csv', type=str, default='data/Voices-AWS/total_dataset.csv', help='Path to input CSV file')
    parser.add_argument('--root_dir', type=str, default='data/Voices-AWS', help='Base path for media files')
    parser.add_argument('--output_dir', type=str, default='data/clips', help='Directory to save output clips and labels')
    parser.add_argument('--clip_duration', type=int, default=5, help='Duration of each clip in seconds')
    parser.add_argument('--overlap', type=int, default=2, help='Overlap duration between clips in seconds')
    parser.add_argument('--max_workers', type=int, default=23, help='Number of threads to use for processing')
    
    return parser.parse_args()

def main():
    args = parse_args()
    csv_path = os.path.join(args.root_dir, 'total_dataset.csv')
    df = pd.read_csv(csv_path)
    # conver start and end times to seconds
    df['start'] = df['start'] / 1000
    df['end'] = df['end'] / 1000
    
    labels_df = create_overlapping_clips(
        df, 
        base_path=args.root_dir,
        output_dir=args.output_dir,
        max_workers=args.max_workers,
        clip_duration=args.clip_duration,
        overlap=args.overlap
    )
    
    print(f"Created {len(labels_df['clip_id'].unique())} clips")
    print(f"Labels saved for {len(labels_df['annotator'].unique())} annotators")

if __name__ == "__main__":
    main()