import os
import pandas as pd
import numpy as np
from pydub import AudioSegment
import av  # <-- Use PyAV instead of OpenCV for video
import uuid
import hashlib  # <-- Add for deterministic clip IDs
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm  # Change to tqdm.auto for better notebook compatibility

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
    """
    Load segments to be excluded from processing.
    
    Args:
        exclusions_path: Path to the exclusions CSV file
        
    Returns:
        Dict mapping media_file to list of (start, end) tuples
    """
    exclusions = {}
    if os.path.exists(exclusions_path):
        df = pd.read_csv(exclusions_path)
        for _, row in df.iterrows():
            if row['media_file'] not in exclusions:
                exclusions[row['media_file']] = []
            exclusions[row['media_file']].append((row['start'], row['end']))
    return exclusions

def is_segment_excluded(start_time, end_time, excluded_segments):
    """
    Check if a time segment overlaps with any excluded segments.
    
    Args:
        start_time: Start time of segment to check
        end_time: End time of segment to check
        excluded_segments: List of (start, end) tuples representing excluded segments
        
    Returns:
        Boolean indicating if segment should be excluded
    """
    if excluded_segments is None:
        return False
        
    for ex_start, ex_end in excluded_segments:
        # Check for overlap
        if not (end_time <= ex_start or start_time >= ex_end):
            return True
    return False

# Add global caches
audio_cache = {}
video_cache = {}

def get_audio_segment(video_path):
    """
    Retrieve AudioSegment from cache or load if not present.
    """
    if video_path not in audio_cache:
        audio_segment = AudioSegment.from_file(video_path)
        audio_cache[video_path] = audio_segment.set_frame_rate(16000)  # Ensure consistent frame rate
    return audio_cache[video_path]

def get_video_capture(video_path):
    """
    Retrieve cv2.VideoCapture from cache or load if not present.
    """
    if video_path not in video_cache:
        video_cache[video_path] = cv2.VideoCapture(video_path)
    return video_cache[video_path]

def process_clip(args):
    """
    Process a single clip - helper function for parallel processing.
    Uses PyAV for video processing.
    """
    start_time, end_time, video_path, output_dir, clip_duration, clip_id = args

    try:
        # Use cached audio
        audio = get_audio_segment(video_path)
        audio_clip = audio.set_frame_rate(16000)[int(start_time*1000):int(end_time*1000)]
        audio_output_dir = os.path.join(output_dir, 'audios')
        audio_clip.export(os.path.join(audio_output_dir, f'{clip_id}.wav'), format="wav")

        # Use PyAV for video processing
        video_output_dir = os.path.join(output_dir, 'videos')
        os.makedirs(video_output_dir, exist_ok=True)
        output_video_path = os.path.join(video_output_dir, f'{clip_id}.mp4')

        with av.open(video_path) as container:
            video_stream = next(s for s in container.streams if s.type == 'video')
            fps = float(video_stream.average_rate)
            width = video_stream.codec_context.width
            height = video_stream.codec_context.height

            # Seek to the start time (in microseconds)
            container.seek(int(start_time * av.time_base * 1e6), any_frame=False, stream=video_stream)

            # Prepare output container
            output_container = av.open(output_video_path, mode='w')
            out_stream = output_container.add_stream('mpeg4', rate=fps)
            out_stream.width = width
            out_stream.height = height
            out_stream.pix_fmt = 'yuv420p'

            frames_to_write = int(clip_duration * fps)
            frames_written = 0

            for frame in container.decode(video_stream):
                if frame.pts is None:
                    continue
                frame_time = float(frame.pts * video_stream.time_base)
                if frame_time < start_time:
                    continue
                if frame_time >= end_time or frames_written >= frames_to_write:
                    break
                # Write frame to output
                packet = out_stream.encode(frame)
                if packet:
                    output_container.mux(packet)
                frames_written += 1

            # Flush encoder
            for packet in out_stream.encode():
                output_container.mux(packet)
            output_container.close()

        return clip_id, True

    except Exception as e:
        print(f"Error processing clip: {str(e)}")
        return None, False

def deterministic_clip_id(media_file, start_time, end_time, clip_duration):
    """
    Generate a deterministic, reusable clip ID based on media_file, start_time, end_time, and clip_duration.
    """
    id_str = f"{media_file}_{start_time:.3f}_{end_time:.3f}_{clip_duration}"
    return hashlib.md5(id_str.encode('utf-8')).hexdigest()

def process_video_clips(args):
    """
    Process all clips for a single video file in one thread, including audio.
    Uses OpenCV for video processing.
    """
    import cv2  # Local import to avoid global dependency if not needed elsewhere

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

    # if output dir not empty return early
    # if os.path.exists(video_output_dir) and os.listdir(video_output_dir):
    #     return video_clips, set([clip_id for clip_id, _, _ in video_clips])
    
    # Process video and audio with OpenCV and pydub in the same thread
    try:
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        for clip_id, start_time, end_time in video_clips:
            # Audio (process and export within this thread)
            audio_clip = audio[int(start_time*1000):int(end_time*1000)]
            audio_clip.export(os.path.join(audio_output_dir, f'{clip_id}.wav'), format="wav")

            # Video
            output_video_path = os.path.join(video_output_dir, f'{clip_id}.mp4')
            out_video = cv2.VideoWriter(
                output_video_path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                (width, height)
            )

            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)
            video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frames_written = 0
            total_frames = end_frame - start_frame

            for _ in range(total_frames):
                ret, frame = video.read()
                if not ret:
                    break
                out_video.write(frame)
                frames_written += 1

            out_video.release()
            successful_clips.add(clip_id)
        video.release()
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
        futures = [executor.submit(process_video_clips, args) for args in video_args]
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
    label_cols = ['SR', 'ISR', 'MUR', 'P', 'B', 'V', 'FG', 'HM', 'ME', 'T']

    for annotator in tqdm(labels_df['annotator'].unique(), desc="Saving annotator labels"):
        annotator_df = labels_df[labels_df['annotator'] == annotator]
        gold_df = labels_df[labels_df['annotator'] == 'Gold']

        # For test split, use gold labels if available
        if not gold_df.empty:
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
        annotator_df.to_csv(os.path.join(output_dir, 'labels', f'{annotator}_labels.csv'), index=False)
        
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
    csv_path = os.path.join(args.root_dir, '_tmp_total_dataset.csv')
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