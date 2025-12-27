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
from peft import LoraConfig, get_peft_model
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


def train_model(dataset_dir: str, output_dir: str, config: dict, model_name: str = None, gui_mode: bool = False):
    """
    Fine-tune image classification model.

    Args:
        dataset_dir: Directory with train/val/test splits
        output_dir: Where to save model checkpoints
        config: Configuration dictionary
        model_name: HuggingFace model ID (overrides config)
        gui_mode: Enable GUI-friendly output (less verbose)
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

    # Apply PEFT (LoRA) if configured
    use_peft = config['training'].get('use_peft', False)
    if use_peft:
        print("\nApplying PEFT (LoRA) for parameter-efficient fine-tuning...")
        peft_config = config['training']['peft']

        # Scan model to show all available linear layer types
        print("Scanning model architecture...")

        # Get all unique linear layer names in the model
        all_linear_names = set()
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                parts = name.split('.')
                if len(parts) > 0:
                    all_linear_names.add(parts[-1])

        print(f"\nAvailable linear layers for LoRA targeting:")
        for layer_name in sorted(all_linear_names):
            print(f"  - {layer_name}")

        # Get configured target modules
        target_modules = peft_config.get('target_modules')

        if not target_modules:
            raise ValueError(
                "No target_modules specified in config.yaml!\n"
                f"Available modules: {sorted(all_linear_names)}\n"
                "Please set training.peft.target_modules in config.yaml"
            )

        # Validate that configured modules actually exist
        invalid_modules = [m for m in target_modules if m not in all_linear_names]
        if invalid_modules:
            raise ValueError(
                f"Invalid target_modules in config: {invalid_modules}\n"
                f"Available modules: {sorted(all_linear_names)}\n"
                "Please update training.peft.target_modules in config.yaml"
            )

        print(f"\nConfigured target modules: {target_modules}")
        print(f"✓ All target modules are valid\n")

        lora_config = LoraConfig(
            r=peft_config['r'],
            lora_alpha=peft_config['lora_alpha'],
            lora_dropout=peft_config['lora_dropout'],
            target_modules=target_modules,
            bias=peft_config['bias'],
            modules_to_save=['classifier'],  # Always train the classifier head
        )

        model = get_peft_model(model, lora_config)

        # Verify trainable parameters
        print("\nTrainable parameter breakdown:")
        model.print_trainable_parameters()

        # Show which modules got LoRA adapters
        print("\nModules with LoRA adapters:")
        lora_modules = []
        for name, module in model.named_modules():
            if 'lora' in name.lower():
                lora_modules.append(name)

        if lora_modules:
            for mod in lora_modules[:5]:  # Show first 5
                print(f"  - {mod}")
            if len(lora_modules) > 5:
                print(f"  ... and {len(lora_modules) - 5} more")
        else:
            print("  ⚠ WARNING: No LoRA modules found!")
            print("  This means LoRA adapters were not applied correctly.")
            print("  The model will not train properly.")

        # Double-check that gradients are enabled
        trainable_params_with_grad = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_params_with_grad.append(name)

        print(f"\nTotal parameters requiring gradients: {len(trainable_params_with_grad)}")
        if len(trainable_params_with_grad) > 0:
            print("Sample trainable parameters:")
            for name in trainable_params_with_grad[:3]:
                print(f"  - {name}")
        print()
    else:
        print("\nUsing full fine-tuning (all parameters trainable)")
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable params: {trainable_params:,} / {all_params:,} (100.00%)")
        print()

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
        logging_steps=10 if not gui_mode else 1000,  # Less frequent logging in GUI mode
        logging_strategy="steps",
        save_total_limit=3,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to="none",  # Disable TensorBoard
        dataloader_num_workers=4,
        fp16=torch.cuda.is_available(),
        disable_tqdm=gui_mode,  # Disable progress bars in GUI mode
        log_level="warning" if gui_mode else "info",  # Less verbose in GUI mode
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
    print("\n" + "=" * 80)
    print("TRAINING STARTED")
    print("=" * 80)
    if gui_mode:
        print(f"Training for up to {config['training']['epochs']} epochs...")
        print(f"Dataset: {len(dataset['train'])} train, {len(dataset['validation'])} val, {len(dataset['test'])} test")
        print()

    train_result = trainer.train()

    if gui_mode:
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        print(f"Total epochs: {train_result.metrics.get('epoch', 0):.0f}")
        print(f"Training time: {train_result.metrics.get('train_runtime', 0):.1f}s")
        print(f"Final loss: {train_result.metrics.get('train_loss', 0):.4f}")
        print()

    # Evaluate on test set
    print("=" * 80)
    print("EVALUATING ON TEST SET")
    print("=" * 80)
    test_results = trainer.evaluate(dataset['test'])

    print("\nTest Results:")
    for key, value in test_results.items():
        print(f"  {key}: {value:.4f}")

    # Check model performance and give feedback
    test_f1 = test_results.get('eval_f1', 0)
    test_acc = test_results.get('eval_accuracy', 0)

    print("\n" + "=" * 80)
    if test_f1 < 0.7 or test_acc < 0.7:
        print("⚠ WARNING: Model performance is low!")
        print()
        if len(dataset['train']) < 500:
            print(f"  Your training set has only {len(dataset['train'])} samples.")
            print("  Recommendation: Label at least 500-1000 frames for better results.")
        print()
        print("  Tips to improve performance:")
        print("  - Label more frames from diverse sections of videos")
        print("  - Ensure balanced classes (50/50 touching vs not_touching)")
        print("  - Include challenging examples (hand near face, scratching, etc.)")
        print("  - Label frames from different camera angles and lighting")
    else:
        print("✓ Model performance looks good!")
    print("=" * 80)

    # Save final model
    final_model_dir = os.path.join(output_dir, 'final')

    if use_peft:
        # For PEFT models, save only the adapter
        print("\nSaving PEFT model...")
        print(f"  Output directory: {final_model_dir}")

        # Save adapter
        print("  Saving LoRA adapter...")
        model.save_pretrained(final_model_dir)

        # Verify adapter files were created
        adapter_files = ['adapter_config.json', 'adapter_model.safetensors', 'adapter_model.bin']
        found_adapters = [f for f in adapter_files if os.path.exists(os.path.join(final_model_dir, f))]
        if found_adapters:
            print(f"    ✓ Adapter files created: {found_adapters}")
        else:
            print(f"    ⚠ WARNING: No adapter files found! Expected one of: {adapter_files}")

        # Save the base model ID so we know which model to load during inference
        base_model_id_path = os.path.join(final_model_dir, 'base_model_id.txt')
        print(f"  Saving base model ID: {model_id}")
        with open(base_model_id_path, 'w') as f:
            f.write(model_id)
        print(f"    ✓ Base model ID saved")

        # Save marker file
        marker_path = os.path.join(final_model_dir, 'is_peft_model')
        with open(marker_path, 'w') as f:
            f.write('true')
        print(f"    ✓ PEFT marker created")

        # List all saved files for verification
        print(f"\n  Final directory contents:")
        for item in sorted(os.listdir(final_model_dir)):
            item_path = os.path.join(final_model_dir, item)
            if os.path.isfile(item_path):
                size = os.path.getsize(item_path)
                print(f"    - {item} ({size:,} bytes)")
            else:
                print(f"    - {item}/ (directory)")

    else:
        # For full fine-tuning, save normally
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
    parser.add_argument('--gui-mode', action='store_true', help='Enable GUI-friendly output (less verbose)')

    args = parser.parse_args()

    # Set environment variable for GUI mode to suppress progress bars
    if args.gui_mode:
        os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

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
    results = train_model(args.dataset, output_dir, config, args.model, args.gui_mode)


if __name__ == '__main__':
    main()
