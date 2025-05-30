import argparse
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from torch.utils.data import DataLoader
from configs import from_args
from models import  prep_model
from dataset import prep_dataset, collate_fn

import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Train a stuttering detection model")
    # dataset arguments
    parser.add_argument('--dataset_root', type=str, default='data/clips', help='Root directory for dataset')
    parser.add_argument('--dataset_annotator', type=str, default='bau', help='Path to the label file')
    parser.add_argument('--dataset_label', type=str, default=None, help='Label to use for upsampling')
    parser.add_argument('--dataset_sampling_rate', type=int, default=16000, help='Sampling rate for audio')
    # audio model arguments 
    parser.add_argument('--audio_pretrained_model_name', type=str, default='facebook/wav2vec2-base-960h', help='Pretrained audio model name')
    parser.add_argument('--audio_freeze_encoder', type=bool, default=True, help='Freeze encoder layers of audio model')
    parser.add_argument('--audio_freeze_feature_extractor', type=bool, default=True, help='Freeze feature extractor layers of audio model')
    parser.add_argument('--audio_unfreeze_layers', type=int, default=None, help='Unfreeze layers of audio model')
    parser.add_argument('--audio_learning_rate', type=float, default=5e-5, help='Learning rate for audio model')
    # video model arguments
    parser.add_argument('--video_pretrained_model_name', type=str, default='google/vivit-b-16x2-kinetics400', help='Pretrained video model name')
    parser.add_argument('--video_freeze_backbone', type=bool, default=True, help='Freeze backbone layers of video model')
    parser.add_argument('--video_unfreeze_last_n_layers', type=int, default=2, help='Unfreeze last N layers of video model')
    parser.add_argument('--video_learning_rate', type=float, default=1e-5, help='Learning rate for video model')
    # multimodal model arguments
    parser.add_argument('--fusion_method', type=str, default='concat', choices=['concat', 'attention'], help='Fusion method for multimodal model')
    parser.add_argument('--fusion_dim', type=int, default=512, help='Fusion dimension for multimodal model')

    # training arguments
    parser.add_argument('--seed', type=int, default=42, help='Random seed for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--max_epochs', type=int, default=30, help='Maximum number of epochs for training')
    parser.add_argument('--gradient_clip_val', type=float, default=1.0, help='Gradient clipping value')

    parser.add_argument('--modality', type=str, choices=['audio', 'video', 'multimodal'], required=True, help='Modality to train on')
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory for saving models and logs')

    return parser.parse_args()



def main(args):
    config = from_args(args)
    # pl.seed_everything(config.seed)
    
    # Initialize logger
    # logger = WandbLogger(name="stuttering-detection", project="stuttering-detection", log_model=True, config=config)
    logger = TensorBoardLogger(os.path.join(config.output_dir, 'tb_logs'), name="{}-stuttering-detection".format(config.modality))
    # logger.log_hyperparams(vars(config))

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(config.output_dir, 'checkpoints'),
        filename='stuttering-{epoch:02d}-{val_loss:.2f}',
        monitor='val/loss',
        mode='min',
        save_top_k=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val/loss',
        patience=10,
        mode='min'
    )
    
    train_dataset = prep_dataset(config.dataset_config, split='train', modality=config.modality)
    # train_size = int(0.8 * len(dataset))
    # val_size = len(dataset) - train_size
    # train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn
    )
    val_dataset = prep_dataset(config.dataset_config, split='val', modality=config.modality)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn
    )

    test_dataset = prep_dataset(config.dataset_config, split='test', modality=config.modality)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn
    )

    model = prep_model(config)
    # Trainer
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        accelerator='gpu',
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping],
        gradient_clip_val=1.0
    )
    # Train
    trainer.fit(model, train_loader, val_loader)

    # Test
    trainer.test(model, test_loader)

if __name__ == "__main__":
    args = parse_args()
    main(args)
