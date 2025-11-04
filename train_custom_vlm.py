"""
Training Script for Custom Vision-Language Model
Fine-tune on Flickr8k dataset for image captioning
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image
import os
from tqdm import tqdm
import json
from custom_vlm import VisionLanguageModel, CustomVLMProcessor


class Flickr8kDataset(Dataset):
    """
    Dataset class for Flickr8k
    """

    def __init__(self, captions_file, images_dir, processor, max_length=128):
        self.images_dir = images_dir
        self.processor = processor
        self.max_length = max_length

        # Load captions
        self.data = []
        with open(captions_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                # Try different delimiters: comma, pipe, tab
                parts = None
                if ',' in line:
                    parts = line.split(',', 1)
                elif '|' in line:
                    parts = line.split('|', 1)
                elif '\t' in line:
                    parts = line.split('\t', 1)

                if parts is None or len(parts) != 2:
                    continue

                image_name, caption = parts
                image_name = image_name.strip()
                caption = caption.strip()

                image_path = os.path.join(images_dir, image_name)
                if os.path.exists(image_path):
                    self.data.append({
                        'image_path': image_path,
                        'caption': caption
                    })

        print(f"Loaded {len(self.data)} image-caption pairs")

        if len(self.data) == 0:
            raise ValueError(f"No valid image-caption pairs found! Check that:\n"
                           f"1. Captions file exists: {captions_file}\n"
                           f"2. Images directory exists: {images_dir}\n"
                           f"3. Image files match the names in captions file")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Load and process image
        try:
            image = Image.open(item['image_path']).convert('RGB')
        except Exception as e:
            print(f"Error loading image {item['image_path']}: {e}")
            # Return a dummy image if loading fails
            image = Image.new('RGB', (224, 224), color='gray')

        caption = item['caption']

        # Process image and text
        encoding = self.processor(
            images=image,
            text=caption,
            return_tensors="pt",
            padding='max_length',
            truncation=True,
            max_length=self.max_length
        )

        # Remove batch dimension
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}

        # Add labels (same as input_ids for language modeling)
        encoding['labels'] = encoding['input_ids'].clone()

        return encoding


def collate_fn(batch):
    """
    Custom collate function to batch samples
    """
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])

    return {
        'pixel_values': pixel_values,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }


def train_epoch(model, dataloader, optimizer, device, epoch, total_epochs):
    """
    Train for one epoch
    """
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs}")

    for batch_idx, batch in enumerate(progress_bar):
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Update metrics
        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)

        progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})

    return total_loss / len(dataloader)


def validate(model, dataloader, device):
    """
    Validate the model
    """
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()

    return total_loss / len(dataloader)


def train_custom_vlm(
    captions_file="captions.txt",
    images_dir="Images",
    output_dir="custom_vlm_finetuned",
    num_epochs=20,
    batch_size=16,
    learning_rate=5e-5,
    warmup_ratio=0.1,
    save_every=5
):
    """
    Main training function
    """
    print("="*70)
    print("CUSTOM VLM TRAINING")
    print("="*70)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize model and processor
    print("\nInitializing model...")
    model = VisionLanguageModel()
    model = model.to(device)
    processor = CustomVLMProcessor()

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Load dataset
    print("\nLoading dataset...")
    dataset = Flickr8kDataset(captions_file, images_dir, processor)

    # Split dataset (90% train, 10% val)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for Windows
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True
    )

    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    num_training_steps = len(train_loader) * num_epochs
    num_warmup_steps = int(warmup_ratio * num_training_steps)

    scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps - num_warmup_steps)

    # Training loop
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 70)

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch, num_epochs)
        train_losses.append(train_loss)

        # Validate
        val_loss = validate(model, val_loader, device)
        val_losses.append(val_loss)

        # Update scheduler
        scheduler.step()

        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(output_dir, "pytorch_model.bin")
            torch.save(model.state_dict(), best_model_path)
            print(f"  Saved best model (val_loss: {val_loss:.4f})")

        # Save checkpoint
        if epoch % save_every == 0:
            checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"  Saved checkpoint at epoch {epoch}")

    # Save final model
    final_model_path = os.path.join(output_dir, "pytorch_model_final.bin")
    torch.save(model.state_dict(), final_model_path)

    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss
    }
    with open(os.path.join(output_dir, "training_history.json"), 'w') as f:
        json.dump(history, f, indent=2)

    print("\n" + "="*70)
    print("TRAINING COMPLETED")
    print("="*70)
    print(f"\nBest validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {output_dir}")

    return model, history


if __name__ == "__main__":
    # Train the model
    train_custom_vlm(
        captions_file="captions.txt",
        images_dir="Images",
        output_dir="custom_vlm_finetuned",
        num_epochs=20,
        batch_size=8,  # Adjust based on GPU memory
        learning_rate=5e-5,
        save_every=5
    )
