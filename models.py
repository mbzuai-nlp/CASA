import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import Wav2Vec2Model, VivitModel
from typing import Dict, List, Optional, Union, Tuple, Any
import torchmetrics
from abc import ABC, abstractmethod

from configs import AudioModelConfig, VideoModelConfig, MultimodalModelConfig, BaseModelConfig, TrainingConfig
from huggingface_hub import hf_hub_download

def _masked_mean(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Compute mean pooling with attention mask"""
    # Extend mask to same shape as hidden states
    mask = attention_mask.unsqueeze(-1).expand(hidden_states.size())
    masked_sum = torch.sum(hidden_states * mask, dim=1)
    mask_sum = torch.sum(mask, dim=1)
    mask_sum = torch.clamp(mask_sum, min=1e-9)
    
    return masked_sum / mask_sum

class BaseModule(pl.LightningModule, ABC):

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """
        Load a model from the Hugging Face Hub.
        
        Args:
            pretrained_model_name_or_path (str): Model name or path.
            **kwargs: Additional arguments for the model.
        
        Returns:
            An instance of the model class.
        """
        # Download the model files
        model_ckpt = hf_hub_download(
            repo_id=pretrained_model_name_or_path,
            filename="model_best.ckpt",
            subfolder="checkpoint",
        )
        model = cls.load_from_checkpoint(model_ckpt, **kwargs)

        return model
 
    def __init__(self, config: BaseModelConfig):
        super().__init__()
        self.config = config
        # self.save_hyperparameters(vars(config))
        
        # Set label names
        self.label_names = config.label_names
        
        # These will be initialized by child classes
        self.backbone = None
        self.classifier = None
        
        self.train_accuracy = torchmetrics.Accuracy(task='binary', num_labels=len(config.label_names))
        self.val_accuracy = torchmetrics.Accuracy(task='binary', num_labels=len(config.label_names))
        self.test_accuracy = torchmetrics.Accuracy(task='binary', num_labels=len(config.label_names))
        
        self.train_f1 = torchmetrics.F1Score(task='binary', num_labels=len(config.label_names))
        self.val_f1 = torchmetrics.F1Score(task='binary', num_labels=len(config.label_names))
        self.test_f1 = torchmetrics.F1Score(task='binary', num_labels=len(config.label_names))

        self.per_class_metrics = nn.ModuleDict()
        for label in self.label_names:
            self.per_class_metrics[f"val/f1_{label}"] = torchmetrics.F1Score(task='binary', num_classes=1)
            self.per_class_metrics[f"test/f1_{label}"] = torchmetrics.F1Score(task='binary', num_classes=1)

        
        self.learning_rate = config.learning_rate
        self.weight_decay = config.weight_decay
        self.scheduler_eta_min = config.scheduler_eta_min
    
    @abstractmethod
    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        pass
    
    def print_trainable_parameters(self):
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Trainable parameters: {trainable_params:,} of {total_params:,} "
              f"({100 * trainable_params / total_params:.2f}%)")
    
    def _get_labels_tensor(self, batch: Dict) -> torch.Tensor:
        """Convert batch labels to a tensor"""
        batch_size = len(batch["clip_id"])
        labels = torch.zeros(batch_size, len(self.label_names), device=self.device)
        for i, label in enumerate(self.label_names):
            if label in batch:
                # Convert label values to float
                label_values = batch[label]
                for j, val in enumerate(label_values):
                    if val is not None:
                        labels[j, i] = float(val)
        return labels
    
    def _common_step(self, batch, batch_idx, step_type):
        """Common step for training, validation and testing"""
        # Get inputs and labels based on child implementation
        inputs, labels = self._prepare_batch(batch)
        
        # Forward pass
        outputs = self(**inputs)
        logits = outputs["logits"]
        
        # Compute loss
        loss_weights = None
        if self.config.loss_weights is not None:
            loss_weights = torch.tensor(self.config.loss_weights, device=self.device)
            
        loss = F.binary_cross_entropy_with_logits(logits, labels, loss_weights, reduction='mean')
        
        # Compute predictions
        preds = torch.sigmoid(logits) > 0.5

        # Update metrics
        if step_type == "train":
            self.train_accuracy(preds, labels)
            self.train_f1(preds, labels)
            self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
            self.log("train/acc", self.train_accuracy, prog_bar=True, on_epoch=True)
            self.log("train/f1", self.train_f1, prog_bar=True, on_epoch=True)
        elif step_type == "val":
            self.val_accuracy(preds, labels)
            self.val_f1(preds, labels)
            self.log("val/loss", loss, prog_bar=True, on_epoch=True)
            self.log("val/acc", self.val_accuracy, prog_bar=True, on_epoch=True)
            self.log("val/f1", self.val_f1, prog_bar=True, on_epoch=True)
            self._update_per_class_metrics(preds, labels, step_type)
        elif step_type == "test":
            self.test_accuracy(preds, labels)
            self.test_f1(preds, labels)
            self.log("test/loss", loss, prog_bar=True, on_epoch=True)
            self.log("test/acc", self.test_accuracy, prog_bar=True, on_epoch=True)
            self.log("test/f1", self.test_f1, prog_bar=True, on_epoch=True)
            self._update_per_class_metrics(preds, labels, step_type)
        
        return {
            "loss": loss,
            "preds": preds,
            "labels": labels
        }
    
    def _update_per_class_metrics(self, preds, labels, step_type):
        """Helper method to update per-class metrics"""
        for i, label in enumerate(self.label_names):
            class_preds = preds[:, i]
            class_labels = labels[:, i]
            
            # Skip empty classes (all labels are 0)
            if class_labels.sum() > 0:
                # Update F1
                self.per_class_metrics[f"{step_type}/f1_{label}"](class_preds, class_labels)
                self.log(
                    f"{step_type}/f1_{label}", 
                    self.per_class_metrics[f"{step_type}/f1_{label}"], 
                    on_epoch=True
                )
                
                # # Update Accuracy
                # self.per_class_metrics[f"{step_type}_acc_{label}"](class_preds, class_labels)
                # self.log(
                #     f"{step_type}_acc_{label}", 
                #     self.per_class_metrics[f"{step_type}_acc_{label}"], 
                #     on_epoch=True
                # )

    @abstractmethod
    def _prepare_batch(self, batch) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        pass
    
    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")
    
    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "val")
    
    def test_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "test")
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler"""
        optimizer = AdamW(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = {
            "scheduler": CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.estimated_stepping_batches,
                eta_min=self.scheduler_eta_min
            ),
            "interval": "step",
            "frequency": 1
        }
        
        return [optimizer], [scheduler]
    
    def predict_stuttering(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, Dict]:

        self.eval()
        with torch.no_grad():
            # Forward pass
            outputs = self(**self._prepare_prediction_inputs(inputs))
            logits = outputs["logits"]
            
            # Compute probabilities and predictions
            probs = torch.sigmoid(logits)
            preds = probs > 0.5
        
        results = {}
        for i, label in enumerate(self.label_names):
            results[label] = {
                "probability": probs[0, i].item(),
                "prediction": preds[0, i].item()
            }

        
        return results
    
    @abstractmethod
    def _prepare_prediction_inputs(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pass

class AudioClassificationModule(BaseModule):
    """Wav2Vec2-based stuttering classification model"""

    def __init__(self, config: AudioModelConfig):
        super().__init__(config)
        
        # Load pre-trained wav2vec model
        self.backbone = Wav2Vec2Model.from_pretrained(config.pretrained_model_name)
        
        # Get hidden size from config
        hidden_size = self.backbone.config.hidden_size
        
        # Freeze feature extractor if needed
        if config.freeze_feature_extractor:
            self.freeze_feature_extraction()
        
        # Freeze encoder if needed
        if config.freeze_encoder:
            self.freeze_encoder()
            
        # Unfreeze specific layers if requested
        if config.unfreeze_layers:
            self.unfreeze_layers(config.unfreeze_layers)

        # create separate classification heads for each label
        if config.label_names is None or len(config.label_names) == 0:
            raise ValueError("label_names must be provided and non-empty for audio model")
        
        self.classifier = nn.ModuleDict()
        for label in self.label_names:
            self.classifier[label] = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(hidden_size, 1)  # Binary classification for each label
            )
        
        self.print_trainable_parameters()
    
    def freeze_feature_extraction(self):
        """Freeze the feature extraction part of wav2vec2"""
        for param in self.backbone.feature_extractor.parameters():
            param.requires_grad = False
    
    def freeze_encoder(self):
        """Freeze the transformer encoder part of wav2vec2"""
        for param in self.backbone.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_layers(self, layer_ids: List[int]):
        """Unfreeze specific encoder layers for fine-tuning"""
        for layer_id in layer_ids:
            for param in self.backbone.encoder.layers[layer_id].parameters():
                param.requires_grad = True
    
    def forward(
        self, 
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for audio model"""

        outputs = self.backbone(
            input_values=input_values,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.last_hidden_state  # (B, T, D)
        
        # Pool hidden states
        if attention_mask is not None:
            # Masked mean pooling
            pooled = self._masked_mean(hidden_states, attention_mask)
        else:
            # Global mean pooling
            pooled = hidden_states.mean(dim=1)  # (B, D)
        
        # Classification head
        logits = {}
        for label, head in self.classifier.items():
            logits[label] = head(pooled)
        # Convert logits to a single tensor
        logits = torch.cat([logits[label] for label in self.label_names], dim=1) # shape (B, num_labels)
        
        return {
            "logits": logits,
            "pooled_output": pooled,
            "hidden_states": outputs.hidden_states
        }

    def _prepare_batch(self, batch) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Prepare audio inputs and labels from batch"""
        audio_inputs = batch["audio_inputs"] # (B, 1, T)
        input_values = audio_inputs["input_values"].squeeze(1)  # (B, T)
        attention_mask = audio_inputs.get("attention_mask", None)
        
        inputs = {"input_values": input_values}
        if attention_mask is not None:
            inputs["attention_mask"] = attention_mask
            
        labels = self._get_labels_tensor(batch)
        
        return inputs, labels
    
    def _prepare_prediction_inputs(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        input_values = inputs["input_values"].to(self.device)
        
        prepared_inputs = {"input_values": input_values}
        if "attention_mask" in inputs:
            prepared_inputs["attention_mask"] = inputs["attention_mask"].to(self.device)
            
        return prepared_inputs

class VideoClassificationModule(BaseModule):
    """ViVIT-based stuttering classification model"""
    
    def __init__(self, config: VideoModelConfig):
        super().__init__(config)
        
        # Load pre-trained ViVIT model
        self.backbone = VivitModel.from_pretrained(config.pretrained_model_name)
        
        # Get hidden size from config
        hidden_size = self.backbone.config.hidden_size
        
        # Freeze backbone if needed
        if config.freeze_backbone:
            self.freeze_backbone()
            
            # Optionally unfreeze the last N transformer layers
            if config.unfreeze_last_n_layers > 0:
                self.unfreeze_last_n_layers(config.unfreeze_last_n_layers)
        
        self.classifier = nn.ModuleDict()
        for label in config.label_names:
            # Create a separate classification head for each label
            self.classifier[label] = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(hidden_size, 1)  # Binary classification for each label
            )

        
        self.print_trainable_parameters()
    
    def freeze_backbone(self):
        """Freeze the entire ViVIT backbone"""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_last_n_layers(self, n: int):
        """Unfreeze the last N transformer layers of ViVIT"""
        # Unfreeze specific layers (the last n layers)
        for i in range(len(self.backbone.encoder.layer) - n, len(self.backbone.encoder.layer)):
            for param in self.backbone.encoder.layer[i].parameters():
                param.requires_grad = True
    
    def forward(
        self, 
        pixel_values: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for video model"""
        # Get ViVIT outputs
        outputs = self.backbone(
            pixel_values=pixel_values.squeeze(1),  # (B, C, T, H, W) -> (B, T, C, H, W)
            output_hidden_states=True,
        )
        
        # Get the pooler output
        pooled_output = outputs.pooler_output  # [batch_size, hidden_size]
        
        # Classification head
        logits = {}
        for label, head in self.classifier.items():
            logits[label] = head(pooled_output)
        
        logits = torch.cat([logits[label] for label in self.label_names], dim=1)
        
        return {
            "logits": logits,
            # "pooled_output": pooled_output,
            # "hidden_states": outputs.hidden_states
        }
    
    def _prepare_batch(self, batch) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Prepare video inputs and labels from batch"""
        video_inputs = batch["video_inputs"]
        pixel_values = video_inputs["pixel_values"]
        
        inputs = {"pixel_values": pixel_values}
        labels = self._get_labels_tensor(batch)
        
        return inputs, labels
    
    def _prepare_prediction_inputs(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Prepare video inputs for prediction"""
        return {"pixel_values": inputs["pixel_values"].to(self.device)}

class MultimodalClassificationModule(BaseModule):
    """
    Multimodal (audio+video) stuttering detection model
    
    This class combines both Wav2Vec2 and ViVIT models for multimodal analysis.
    """
    
    def __init__(self, config: MultimodalModelConfig):
        # We'll use the video config as the base since it typically has lower learning rate
        super().__init__(config)
        
        # Save both configs
        self.audio_config = config.audio_config
        self.video_config = config.video_config
        self.save_hyperparameters(vars(config))
        
        # Load pre-trained models
        self.audio_backbone = Wav2Vec2Model.from_pretrained(config.audio_config.pretrained_model_name)
        self.video_backbone = VivitModel.from_pretrained(config.video_config.pretrained_model_name)
        
        # Get hidden sizes
        audio_hidden_size = self.audio_backbone.config.hidden_size
        video_hidden_size = self.video_backbone.config.hidden_size
        combined_hidden_size = audio_hidden_size + video_hidden_size
        
        # Apply freezing strategies
        if config.audio_config.freeze_feature_extractor:
            for param in self.audio_backbone.feature_extractor.parameters():
                param.requires_grad = False
                
        if config.audio_config.freeze_encoder:
            for param in self.audio_backbone.encoder.parameters():
                param.requires_grad = False
                
        if config.video_config.freeze_backbone:
            for param in self.video_backbone.parameters():
                param.requires_grad = False
            
        # Selective unfreezing
        if config.audio_config.unfreeze_layers:
            for layer_id in config.audio_config.unfreeze_layers:
                for param in self.audio_backbone.encoder.layers[layer_id].parameters():
                    param.requires_grad = True
                    
        if config.video_config.unfreeze_last_n_layers > 0:
            for i in range(len(self.video_backbone.encoder.layer) - config.video_config.unfreeze_last_n_layers, 
                           len(self.video_backbone.encoder.layer)):
                for param in self.video_backbone.encoder.layer[i].parameters():
                    param.requires_grad = True
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(combined_hidden_size, combined_hidden_size),
            nn.LayerNorm(combined_hidden_size),
            nn.GELU(),
            nn.Dropout(config.video_config.dropout)
        )
        
        # Classification head
        self.classifier = nn.ModuleDict()
        for label in config.label_names:
            self.classifier[label] = nn.Sequential(
                nn.Linear(combined_hidden_size, combined_hidden_size // 2),
                nn.LayerNorm(combined_hidden_size // 2),
                nn.GELU(),
                nn.Dropout(config.video_config.dropout),
                nn.Linear(combined_hidden_size // 2, 1)
            )
        
        # Print parameter stats
        self.print_trainable_parameters()
    
    def forward(
        self, 
        input_values: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for multimodal model"""
        # Process audio
        audio_outputs = self.audio_backbone(
            input_values=input_values,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Process video
        video_outputs = self.video_backbone(
            pixel_values=pixel_values,
            output_hidden_states=True
        )
        
        # Get audio pooled representation
        audio_hidden_states = audio_outputs.last_hidden_state
        # if attention_mask is not None:
        #     # Masked mean pooling
        #     audio_pooled = self._masked_mean(audio_hidden_states, attention_mask)
        # else:
        # Global mean pooling
        audio_pooled = audio_hidden_states.mean(dim=1)
    
        # Get video pooled representation
        video_pooled = video_outputs.pooler_output
        
        # Concatenate modalities
        multimodal_features = torch.cat([audio_pooled, video_pooled], dim=1)
        
        # Fusion
        fused_features = self.fusion(multimodal_features)
        
        # Classification
        logits = {}
        for label, head in self.classifier.items():
            logits[label] = head(fused_features)
        # Convert logits to a single tensor
        logits = torch.cat([logits[label] for label in self.label_names], dim=1)
        
        return {
            "logits": logits,
            # "pooled_output": fused_features,
            # "audio_pooled": audio_pooled,
            # "video_pooled": video_pooled,
            # "audio_hidden_states": audio_outputs.hidden_states,
            # "video_hidden_states": video_outputs.hidden_states
        }
    
    def _prepare_batch(self, batch) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Prepare multimodal inputs and labels from batch"""

        # Prepare inputs dict
        inputs = {
            "input_values": batch["audio_inputs"]["input_values"].squeeze(1),  # (B, T)
            "pixel_values": batch["video_inputs"]["pixel_values"].squeeze(1)  # (B, T, C, H, W)
        }
        
        inputs["attention_mask"] = batch["audio_inputs"].get(["attention_mask"], None)
            
        labels = self._get_labels_tensor(batch)
        
        return inputs, labels
    
    def _prepare_prediction_inputs(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Prepare multimodal inputs for prediction"""
        prepared_inputs = {
            "input_values": inputs["audio_inputs"]["input_values"].to(self.device),
            "pixel_values": inputs["video_inputs"]["pixel_values"].to(self.device)
        }
        
        if "attention_mask" in inputs["audio_inputs"]:
            prepared_inputs["attention_mask"] = inputs["audio_inputs"]["attention_mask"].to(self.device)
            
        return prepared_inputs

def prep_model(config: TrainingConfig) -> BaseModule:

    if config.modality == "audio":
        config.audio_model_config.label_names = ['P', 'B', 'SR', 'ISR','MUR', 'any']
        return AudioClassificationModule(config.audio_model_config)
    elif config.modality == "video":
        config.video_model_config.label_names = ['FG', 'HM', 'V', 'any']
        return VideoClassificationModule(config.video_model_config)
    elif config.modality == "multimodal":
        model_config = MultimodalModelConfig(
            audio_config=config.audio_model_config,
            video_config=config.video_model_config,
            fusion_method=config.fusion_method,
            fusion_dim=config.fusion_dim
        )
        model_config.label_names = ['P', 'B', 'SR', 'ISR', 'MUR', 'FG', 'HM', 'V', 'any']
        return MultimodalClassificationModule(model_config)
    else:
        raise ValueError(f"Unsupported modality: {config.modality}")
    

if __name__ == "__main__":

    audio_config = AudioModelConfig()
    video_config = VideoModelConfig()
    multimodal_config = MultimodalModelConfig(audio_config=audio_config, video_config=video_config)
    
    training_config = TrainingConfig(
        modality="multimodal",
        audio_model_config=audio_config,
        video_model_config=video_config,
        fusion_method="concat",
        fusion_dim=512
    )
    

    model = prep_model(training_config)
    print(model)
    # dummy input for testing
    dummy_audio_input = {
        "audio_inputs": {
            "input_values": torch.randn(2, 16000),  # (batch_size, channels, length)
            "attention_mask": torch.ones(2, 16000)  # (batch_size, length)
        },
        "video_inputs": {
            "pixel_values": torch.randn(2, 32, 3, 224, 224)  # (batch_size, channels, height, width)
        }
    }

    outputs = model(**dummy_audio_input["audio_inputs"], **dummy_audio_input["video_inputs"])
    print(outputs["logits"].shape) 