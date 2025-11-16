"""
VideoMAE Model Fine-Tuning Script

This script provides a template for fine-tuning the VideoMAE model
on your own crime detection dataset.

Requirements:
- GPU with 8GB+ VRAM
- Labeled video dataset
- transformers, datasets, accelerate packages

Usage:
    python finetune_videomae.py --dataset_path /path/to/dataset --epochs 10
"""

import torch
import argparse
import logging
from pathlib import Path
from transformers import (
    VideoMAEForVideoClassification,
    VideoMAEImageProcessor,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from datasets import load_dataset, DatasetDict
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune VideoMAE for crime detection")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--model_path", type=str, default="Backend/models/videomae-large-finetuned-UCF-Crime",
                       help="Path to pre-trained model")
    parser.add_argument("--output_dir", type=str, default="./fine-tuned-videomae",
                       help="Output directory for fine-tuned model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per device")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Warmup steps")
    parser.add_argument("--eval_steps", type=int, default=100, help="Evaluation frequency")
    parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint frequency")
    return parser.parse_args()


def compute_metrics(eval_pred):
    """Compute accuracy, precision, recall, and F1 score."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted'
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def main():
    args = parse_args()
    
    logger.info("=" * 60)
    logger.info("VideoMAE Fine-Tuning Script")
    logger.info("=" * 60)
    logger.info(f"Dataset: {args.dataset_path}")
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info(f"Learning Rate: {args.learning_rate}")
    logger.info("=" * 60)
    
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    if device.type == 'cpu':
        logger.warning("⚠️  No GPU detected! Training will be very slow.")
        logger.warning("⚠️  Consider using Google Colab or a GPU instance.")
    
    # Load dataset
    logger.info("Loading dataset...")
    try:
        # Assuming dataset is in ImageFolder format
        # You may need to customize this based on your dataset structure
        dataset = load_dataset(
            "imagefolder",
            data_dir=args.dataset_path,
            split={"train": "train", "validation": "val"}
        )
        logger.info(f"Dataset loaded: {len(dataset['train'])} train, {len(dataset['validation'])} val samples")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        logger.error("Please ensure your dataset is in the correct format:")
        logger.error("  dataset/")
        logger.error("    ├── train/")
        logger.error("    │   ├── class1/")
        logger.error("    │   ├── class2/")
        logger.error("    │   └── ...")
        logger.error("    └── val/")
        logger.error("        ├── class1/")
        logger.error("        ├── class2/")
        logger.error("        └── ...")
        return
    
    # Load model and processor
    logger.info("Loading pre-trained model...")
    try:
        model = VideoMAEForVideoClassification.from_pretrained(
            args.model_path,
            num_labels=len(dataset['train'].features['label'].names),
            ignore_mismatched_sizes=True
        )
        processor = VideoMAEImageProcessor.from_pretrained(args.model_path)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    # Preprocessing function
    def preprocess_function(examples):
        """Preprocess videos for training."""
        videos = [processor(video, return_tensors="pt").pixel_values for video in examples['video']]
        return {
            'pixel_values': torch.stack(videos),
            'labels': examples['label']
        }
    
    # Preprocess datasets
    logger.info("Preprocessing dataset...")
    train_dataset = dataset['train'].map(
        preprocess_function,
        batched=True,
        remove_columns=dataset['train'].column_names
    )
    val_dataset = dataset['validation'].map(
        preprocess_function,
        batched=True,
        remove_columns=dataset['validation'].column_names
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=0.01,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=device.type == 'cuda',  # Use mixed precision on GPU
        dataloader_num_workers=4,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to="tensorboard"
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Train
    logger.info("Starting training...")
    logger.info("=" * 60)
    
    try:
        train_result = trainer.train()
        
        # Save final model
        logger.info("Saving fine-tuned model...")
        trainer.save_model(args.output_dir)
        processor.save_pretrained(args.output_dir)
        
        # Log training results
        logger.info("=" * 60)
        logger.info("Training completed!")
        logger.info(f"Final training loss: {train_result.training_loss:.4f}")
        logger.info(f"Model saved to: {args.output_dir}")
        logger.info("=" * 60)
        
        # Evaluate on validation set
        logger.info("Evaluating on validation set...")
        eval_results = trainer.evaluate()
        
        logger.info("Validation Results:")
        for key, value in eval_results.items():
            logger.info(f"  {key}: {value:.4f}")
        
        logger.info("=" * 60)
        logger.info("✅ Fine-tuning completed successfully!")
        logger.info(f"To use the fine-tuned model, update config.py:")
        logger.info(f"  VIDEOMAE_MODEL_PATH = '{args.output_dir}'")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.error("Check the error message above for details")
        return


if __name__ == "__main__":
    main()
