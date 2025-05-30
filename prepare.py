import os
import pandas as pd
import numpy as np
from pydub import AudioSegment
import av  # <-- Use PyAV instead of OpenCV for video
import uuid
import hashlib  # <-- Add for deterministic clip IDs
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm  # Change to tqdm.auto for better notebook compatibility
import torch
import torchvision
from typing import Dict
from transformers import Wav2Vec2Processor, VivitImageProcessor

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
        audio.numpy().astype(np.float32),  
        sampling_rate=16000,  # Ensure consistent sampling rate
        return_tensors="pt",
    )
    
    return processed

# Add global caches
audio_cache = {}

def get_audio_segment(video_path):
    """
    Retrieve AudioSegment from cache or load if not present.
    """
    if video_path not in audio_cache:
        audio_segment = AudioSegment.from_file(video_path)
        audio_cache[video_path] = audio_segment.set_frame_rate(16000)  # Ensure consistent frame rate
    return audio_cache[video_path]

def deterministic_clip_id(media_file, start_time, end_time, clip_duration):
    """
    Generate a deterministic, reusable clip ID based on media_file, start_time, end_time, and clip_duration.
    """
    id_str = f"{media_file}_{start_time:.3f}_{end_time:.3f}_{clip_duration}"
    return hashlib.md5(id_str.encode('utf-8')).hexdigest()

def process_video(args):
    """
    Process all clips for a single video file in one thread, extracting audio and video features.
    Uses PyAV for video processing and saves preprocessed features directly without saving MP4 files.
    """
    media_file, task, group, excluded_segments, video_path, output_dir, clip_duration, overlap, segment_to_clip_map = args
    
    successful_clips = set()
    audio = get_audio_segment(video_path)
    video_output_dir = os.path.join(output_dir, 'videos')
    audio_output_dir = os.path.join(output_dir, 'audios')
    os.makedirs(video_output_dir, exist_ok=True)
    os.makedirs(audio_output_dir, exist_ok=True)

    # Prepare all clip segments for this video
    duration = audio.duration_seconds
    stride = clip_duration - overlap
    start_times = np.arange(0, duration - clip_duration, stride)
    video_clips = []
    for start_time in start_times:
        end_time = start_time + clip_duration
        if is_segment_excluded(start_time, end_time, excluded_segments):
            continue
        # Use deterministic clip ID
        clip_id = deterministic_clip_id(media_file, start_time, end_time, clip_duration)
        segment_key = (media_file, start_time, end_time)
        segment_to_clip_map[segment_key] = clip_id
        video_clips.append((clip_id, start_time, end_time))
    
    # Process video and audio with PyAV and save preprocessed features
    try:
        with av.open(video_path) as container:
            video_stream = next(s for s in container.streams if s.type == 'video')
            fps = float(video_stream.average_rate)

            for clip_id, start_time, end_time in video_clips:
                # Process and save audio features
                audio_clip = audio[int(start_time*1000):int(end_time*1000)]
                audio_array = torch.tensor(audio_clip.get_array_of_samples())
              
                audio_features = _process_audio(audio_array.unsqueeze(0), audio_processor)
                torch.save(
                    audio_features,
                    os.path.join(audio_output_dir, f'{clip_id}.pt')
                )

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
                torch.save(
                    video_features,
                    os.path.join(video_output_dir, f'{clip_id}.pt')
                )
        
                successful_clips.add(clip_id)
                
    except Exception as e:
        print(f"Error processing video {video_path}: {str(e)}")

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

    print("\nCreating labels...")
    # Create labels for successful clips with progress bar
    for (media_file, task), group in tqdm(grouped, desc="Creating labels"):
        for segment_key, clip_id in segment_to_clip_map.items():
            seg_media_file, start_time, end_time = segment_key
            if seg_media_file != media_file:
                continue
            if clip_id not in all_successful_clips:
                continue
            clip_labels = group[
                ((group['start'] >= start_time) & (group['start'] < end_time)) |
                ((group['end'] > start_time) & (group['end'] <= end_time)) |
                ((group['start'] <= start_time) & (group['end'] >= end_time))
            ]
            for _, row in clip_labels.iterrows():
                clips_metadata.append({
                    'clip_id': clip_id,
                    'media_file': media_file,
                    'task': task,
                    'split': row['split'],
                    'annotator': row['annotator'],
                    'annotation_start': row['start'],
                    'annotation_end': row['end'],
                    'start_time': start_time,
                    'end_time': end_time,
                    'SR': row['SR'],
                    'ISR': row['ISR'], 
                    'MUR': row['MUR'],
                    'P': row['P'],
                    'B': row['B'],
                    'V': row['V'],
                    'FG': row['FG'],
                    'HM': row['HM'],
                    'ME': row['ME'],
                    'T': row['T']
                })

    # Create and save labels DataFrame
    labels_df = pd.DataFrame(clips_metadata)
    all_clip_ids = labels_df['clip_id'].unique()
    meta_cols = ['media_file', 'task', 'split', 'start_time', 'end_time']
    label_cols = ['SR', 'ISR', 'MUR', 'P', 'B', 'V', 'FG', 'HM', 'ME', 'T']
    gold_df = labels_df[labels_df['annotator'] == 'Gold']
    gold_test_df = gold_df[gold_df['split'] == 'test']

    for annotator in tqdm(labels_df['annotator'].unique(), desc="Saving annotator labels"):
        annotator_df = labels_df[labels_df['annotator'] == annotator]
        # gold_df = labels_df[labels_df['annotator'] == 'Gold']

        # For test split, use gold labels if available
        if not gold_test_df.empty:
            annotator_df = annotator_df[annotator_df['split'] != 'test']
            gold_test_df = gold_df[gold_df['split'] == 'test']
            annotator_df = pd.concat([annotator_df, gold_test_df], ignore_index=True)

        # Ensure all clips are present for this annotator
        annotator_clip_ids = set(annotator_df['clip_id'])
        missing_clip_ids = set(all_clip_ids) - annotator_clip_ids
        if missing_clip_ids:
            # Get metadata for missing clips from labels_df (use first row for each clip)
            meta_cols = ['media_file', 'task', 'split', 'start_time', 'end_time']
            meta_df = labels_df.drop_duplicates('clip_id').set_index('clip_id')
            rows = []
            for clip_id in missing_clip_ids:
                meta = meta_df.loc[clip_id]
                row = {col: meta[col] for col in meta_cols}
                row['clip_id'] = clip_id
                row['annotator'] = annotator
                row['annotation_start'] = np.nan
                row['annotation_end'] = np.nan
                for col in label_cols:
                    row[col] = 0
                rows.append(row)
            missing_df = pd.DataFrame(rows)
            annotator_df = pd.concat([annotator_df, missing_df], ignore_index=True)
       
       # Merge/aggregate labels for the same clip
        annotator_df = annotator_df.groupby(['clip_id']).agg(agg_fn()).reset_index()
        annotator_df.to_csv(os.path.join(output_dir, 'labels', f'{annotator}.csv'), index=False)
    
    # Save majority (MAJ) label for each clip
    def majority_vote(x):
        # x is a DataFrame for a single clip_id
        maj = {}
        for col in label_cols:
            # Use mode, fallback to 0 if tie or empty
            vals = x[col].values
            if len(vals) == 0:
                maj[col] = 0
            else:
                counts = np.bincount(vals.astype(int))
                maj[col] = counts.argmax() if counts.sum() > 0 else 0
        return pd.Series(maj)

    maj_df = labels_df[labels_df['annotator'].isin(['A1','A2','A3'])].groupby('clip_id').apply(majority_vote).reset_index()
    maj_df['annotator'] = 'MAJ'
    
    meta_df = labels_df.drop_duplicates('clip_id').set_index('clip_id')
    for col in meta_cols:
        maj_df[col] = maj_df['clip_id'].map(meta_df[col])
    maj_df['annotation_start'] = np.nan
    maj_df['annotation_end'] = np.nan

    # Reorder columns to match annotator files
    maj_df = maj_df[['clip_id', 'media_file', 'task', 'split', 'annotator', 'annotation_start', 'annotation_end', 'start_time', 'end_time'] + label_cols]
    maj_df.to_csv(os.path.join(output_dir, 'labels', 'MAJ.csv'), index=False)

    return labels_df

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Create overlapping clips from dataset")
    # parser.add_argument('--input_csv', type=str, default='data/Voices-AWS/total_dataset.csv', help='Path to input CSV file')
    parser.add_argument('--root_dir', type=str, default='data/Voices-AWS', help='Base path for media files')
    parser.add_argument('--output_dir', type=str, default='data/clips', help='Directory to save output clips and labels')
    parser.add_argument('--clip_duration', type=int, default=5, help='Duration of each clip in seconds')
    parser.add_argument('--overlap', type=int, default=0, help='Overlap duration between clips in seconds')
    parser.add_argument('--max_workers', type=int, default=16, help='Number of threads to use for processing')
    
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