#!/usr/bin/env python3
"""
Train binary classifier for nose-touch detection using HuggingFace Transformers.

Usage:
    python src/train_classifier.py --dataset data/dataset
    python src/train_classifier.py --dataset data/dataset --model facebook/convnext-tiny-224
"""

import os
import argparse
import torch
import numpy as np
from pathlib import Path
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import load_dataset
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from utils import load_config, ensure_dir


def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary'
    )

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def collate_fn(examples):
    """Custom collate function for batching."""
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def train_model(dataset_dir: str, output_dir: str, config: dict, model_name: str = None):
    """
    Fine-tune image classification model.

    Args:
        dataset_dir: Directory with train/val/test splits
        output_dir: Where to save model checkpoints
        config: Configuration dictionary
        model_name: HuggingFace model ID (overrides config)
    """
    # Model configuration
    model_id = model_name if model_name else config['model']['name']
    image_size = config['model']['image_size']

    print(f"Training configuration:")
    print(f"  Model: {model_id}")
    print(f"  Image size: {image_size}")
    print(f"  Dataset: {dataset_dir}")
    print()

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(
        "imagefolder",
        data_dir=dataset_dir,
        drop_labels=False
    )

    print(f"Dataset loaded:")
    print(f"  Train: {len(dataset['train'])} samples")
    print(f"  Validation: {len(dataset['validation'])} samples")
    print(f"  Test: {len(dataset['test'])} samples")
    print()

    # Get label mapping
    labels = dataset['train'].features['label'].names
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}

    print(f"Labels: {labels}")
    print(f"Label mapping: {label2id}")
    print()

    # Calculate class weights if configured
    class_weights = None
    if config['training'].get('use_class_weights', False):
        train_labels = [example['label'] for example in dataset['train']]
        class_counts = np.bincount(train_labels)
        total = len(train_labels)
        class_weights = torch.FloatTensor([total / (len(class_counts) * count) for count in class_counts])
        print(f"Class weights: {class_weights.tolist()}")
        print()

    # Load image processor
    print("Loading image processor...")
    image_processor = AutoImageProcessor.from_pretrained(model_id)

    # Define data augmentation
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std),
    ])

    def preprocess_train(examples):
        """Preprocess training examples with augmentation."""
        examples['pixel_values'] = [train_transforms(image.convert("RGB")) for image in examples['image']]
        return examples

    def preprocess_val(examples):
        """Preprocess validation examples without augmentation."""
        examples['pixel_values'] = [val_transforms(image.convert("RGB")) for image in examples['image']]
        return examples

    # Apply preprocessing
    print("Preprocessing dataset...")
    dataset['train'].set_transform(preprocess_train)
    dataset['validation'].set_transform(preprocess_val)
    dataset['test'].set_transform(preprocess_val)

    # Load model
    print(f"Loading model: {model_id}")
    model = AutoModelForImageClassification.from_pretrained(
        model_id,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print()

    # Training arguments (ensure numeric values are proper types)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=int(config['training']['epochs']),
        per_device_train_batch_size=int(config['training']['batch_size']),
        per_device_eval_batch_size=int(config['training']['batch_size']) * 2,
        learning_rate=float(config['training']['learning_rate']),
        weight_decay=float(config['training']['weight_decay']),
        warmup_steps=int(config['training']['warmup_steps']),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir=os.path.join(output_dir, 'logs'),
        logging_steps=10,
        logging_strategy="steps",
        save_total_limit=3,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to="none",  # Disable TensorBoard
        dataloader_num_workers=4,
        fp16=torch.cuda.is_available(),
    )

    # Custom Trainer class to handle class weights
    class WeightedTrainer(Trainer):
        def __init__(self, *args, class_weights=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.class_weights = class_weights

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits

            if self.class_weights is not None:
                loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
            else:
                loss_fct = torch.nn.CrossEntropyLoss()

            loss = loss_fct(logits, labels)
            return (loss, outputs) if return_outputs else loss

    # Initialize trainer
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config['training']['early_stopping_patience'])],
        class_weights=class_weights
    )

    # Train
    print("Starting training...")
    print("=" * 80)
    trainer.train()

    # Evaluate on test set
    print("\n" + "=" * 80)
    print("Evaluating on test set...")
    test_results = trainer.evaluate(dataset['test'])

    print("\nTest Results:")
    for key, value in test_results.items():
        print(f"  {key}: {value:.4f}")

    # Save final model
    final_model_dir = os.path.join(output_dir, 'final')
    trainer.save_model(final_model_dir)
    image_processor.save_pretrained(final_model_dir)

    print(f"\nModel saved to: {final_model_dir}")
    print("\nTraining complete!")

    return test_results


def main():
    parser = argparse.ArgumentParser(description='Train nose-touch classifier')
    parser.add_argument('--dataset', required=True, help='Dataset directory')
    parser.add_argument('--output', help='Output directory (default: models/deit-small-nose-touch)')
    parser.add_argument('--model', help='HuggingFace model ID (default: from config)')
    parser.add_argument('--config', default='config.yaml', help='Config file path')

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Determine output directory
    if args.output:
        output_dir = args.output
    else:
        model_name = args.model if args.model else config['model']['name']
        safe_name = model_name.replace('/', '-')
        output_dir = os.path.join(config['paths']['models'], f"{safe_name}-nose-touch")

    ensure_dir(output_dir)

    # Train model
    results = train_model(args.dataset, output_dir, config, args.model)


if __name__ == '__main__':
    main()
