<h1 align="center"> Clinical Annoations for Automatic Stuttering Assessment </h1>

<div align="center">

[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-PDF-blue)](https://arxiv.org/abs/2506.00644)
[![Project](https://img.shields.io/badge/Project-CASA-red)](https://github.com/mbzuai-nlp/CASA)
</div>


> **Accepted at Interspeech2025**

This project contains the dataset  and baseline multimodal classification models described by [Paper](https://arxiv.org/abs/2506.00644) Refer  [Annotation Guidline](docs/guidlines.md) to see the details of the annotaion guidlines.


## Project Structure

```
project/
└── data/
    └── Voices-AWS/
        ├── interview/
        │   ├── video/                # Place MP4 files here
        │   ├── total_dataset.csv     # Annotation data
        │   ├── exclusions.csv        # Optional: segments to exclude
        │   └── raw.csv               # Raw data
        └── reading/
            ├── video/                # Place MP4 files here
            ├── total_dataset.csv     # Annotation data
            ├── exclusions.csv        # Optional: segments to exclude
            └── raw.csv    

```

## 📦 Installation

```bash
git clone https://github.com/mbzuai-nlp/CASA.git
cd CASA
conda create -n casa python=3.12
conda activate casa
pip install -r requirements.txt
```

## 📁 Data Preparation

### 1. Prepare Input Data

1. **Download Media Files**: Follow the instructions [Here](docs/download.md)

2. **Verify Input Data Structure**:
   ```
   data/Voices-AWS/interview/
   ├── video/
   │   ├── participant1.mp4
   │   ├── participant2.mp4
   │   └── ...
   ├── total_dataset_final.csv
   └── exclusions.csv
   ```

 ***Required Files***:
   - `total_dataset.csv`: Contains stuttering annotations with columns:
     - media_file: filename without extension
     - item: the group id in the form [media_id-(group_start_time, group_end_time)] after grouping annotations based on region.
     - start: start time in milliseconds
     - end: end time in milliseconds
     - annotator: (A1, A2, A3, Gold) and additional annotator aggrigation methods (BAU, MAS, SAD)
     - SR, ISR, MUR, P, B, V, FG, HM, ME, T: stuttering type indicators (0/1) refere to [Annotation Guidelines](docs/guidlines.md) for details
  - `exclusions.csv` : Containes the unannotated regions. (Interviewer part of interview section)

### 2. Run Dataset Preparation
To prepare the data for training run the following command:
(Note: This takes ~ 30 mins with a 24 core CPU. It also requires >130GB of memory ) 
```bash
python prepare.py \
    --root_dir "/path/to/root/dir" \ 
    --input_dir "/path/to/output/dir" \ 
    --clip_duration 5 \ # duration of each clip in seconds
    --overlap 2 \ # the overlap window
    --max_workers 16 \ # update this number based on the number of cpu cores
```

The script generates:
1. 5 second audio and video features preprocessed using the respective Wav2vec2 and ViViT Processors
2. Labels for each annotator
3. Labels for the aggrigation methods (BAU, MAS, SAD, MAJ)

## 🧠 Models

To train the models, use the following command
```bash
python train.py \
    --modality audio \ # audio, video, multimodal
    --dataset_root "/path/to/dataset/dir" \
    --dataset_annotator "bau" \ #eg annotator to use to train the models
    --output_dir "/path/to/output" \ 
```

## Notes

- Video files should be in MP4 format
- File names in the CSV should match the media files (without extension)
- Start/end times in the CSV should be in milliseconds


## ✏️ Citation

If you find this data annotations helpful, please cite our paper:

```bibtex

```
