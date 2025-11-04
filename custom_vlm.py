"""
Custom Vision-Language Model for Image Captioning
Architecture: Vision Transformer (ViT) + GPT-2 with Cross-Attention
"""

import torch
import torch.nn as nn
from transformers import ViTModel, GPT2LMHeadModel, GPT2Config
from transformers import ViTImageProcessor, GPT2Tokenizer
from PIL import Image
import os


class VisionLanguageModel(nn.Module):
    """
    Custom VLM combining ViT encoder with GPT-2 decoder
    """

    def __init__(self, vision_model_name="google/vit-base-patch16-224",
                 language_model_name="gpt2"):
        super().__init__()

        # Vision Encoder: ViT
        self.vision_encoder = ViTModel.from_pretrained(vision_model_name)
        vision_dim = self.vision_encoder.config.hidden_size  # 768 for ViT-Base

        # Language Decoder: GPT-2
        gpt2_config = GPT2Config.from_pretrained(language_model_name)
        gpt2_config.add_cross_attention = True  # Enable cross-attention
        gpt2_config.is_decoder = True
        self.language_decoder = GPT2LMHeadModel.from_pretrained(
            language_model_name,
            config=gpt2_config
        )
        language_dim = self.language_decoder.config.hidden_size  # 768 for GPT-2

        # Projection layer to align vision and language dimensions
        self.vision_projection = nn.Linear(vision_dim, language_dim)

        # Learnable vision prefix tokens
        self.num_vision_tokens = 32
        self.vision_query = nn.Parameter(torch.randn(1, self.num_vision_tokens, vision_dim))

        # Cross-attention pooling
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=vision_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        # Layer normalization
        self.vision_norm = nn.LayerNorm(language_dim)

    def encode_image(self, pixel_values):
        """
        Encode image using ViT
        Returns: vision embeddings [batch_size, num_tokens, hidden_dim]
        """
        # Get image features from ViT
        vision_outputs = self.vision_encoder(pixel_values=pixel_values)
        image_features = vision_outputs.last_hidden_state  # [B, 197, 768]

        # Use learnable queries with cross-attention to pool image features
        batch_size = image_features.size(0)
        vision_queries = self.vision_query.expand(batch_size, -1, -1)  # [B, 32, 768]

        # Cross-attention: query image features
        attended_features, _ = self.cross_attention(
            vision_queries,
            image_features,
            image_features
        )

        # Project to language dimension
        vision_embeds = self.vision_projection(attended_features)  # [B, 32, 768]
        vision_embeds = self.vision_norm(vision_embeds)

        return vision_embeds

    def forward(self, pixel_values, input_ids=None, attention_mask=None, labels=None):
        """
        Forward pass for training
        """
        # Encode image
        vision_embeds = self.encode_image(pixel_values)  # [B, 32, 768]

        # Get text embeddings
        text_embeds = self.language_decoder.transformer.wte(input_ids)  # [B, seq_len, 768]

        # Concatenate vision and text embeddings
        combined_embeds = torch.cat([vision_embeds, text_embeds], dim=1)  # [B, 32+seq_len, 768]

        # Create attention mask for combined input
        batch_size, seq_len = input_ids.size()
        vision_attention_mask = torch.ones(
            batch_size, self.num_vision_tokens,
            dtype=attention_mask.dtype,
            device=attention_mask.device
        )
        combined_attention_mask = torch.cat([vision_attention_mask, attention_mask], dim=1)

        # Adjust labels to account for vision tokens
        # Prepend -100 (ignore_index) for vision token positions
        if labels is not None:
            vision_labels = torch.full(
                (batch_size, self.num_vision_tokens),
                fill_value=-100,
                dtype=labels.dtype,
                device=labels.device
            )
            combined_labels = torch.cat([vision_labels, labels], dim=1)
        else:
            combined_labels = None

        # Forward through GPT-2 decoder
        outputs = self.language_decoder(
            inputs_embeds=combined_embeds,
            attention_mask=combined_attention_mask,
            labels=combined_labels,
            return_dict=True
        )

        return outputs

    def generate(self, pixel_values, input_ids=None, attention_mask=None, **generation_kwargs):
        """
        Generate captions for inference
        """
        # Encode image
        vision_embeds = self.encode_image(pixel_values)  # [B, 32, 768]

        # If no input_ids, start with BOS token
        batch_size = pixel_values.size(0)
        if input_ids is None:
            # Use empty input for generation
            input_ids = torch.zeros((batch_size, 1), dtype=torch.long, device=pixel_values.device)

        # Get text embeddings
        text_embeds = self.language_decoder.transformer.wte(input_ids)

        # Concatenate vision and text embeddings
        combined_embeds = torch.cat([vision_embeds, text_embeds], dim=1)

        # Create attention mask
        vision_attention_mask = torch.ones(
            batch_size, self.num_vision_tokens,
            dtype=torch.long,
            device=pixel_values.device
        )
        if attention_mask is None:
            text_attention_mask = torch.ones_like(input_ids)
        else:
            text_attention_mask = attention_mask

        combined_attention_mask = torch.cat([vision_attention_mask, text_attention_mask], dim=1)

        # Generate using GPT-2
        # Note: We need to handle this carefully since we're using embeddings
        outputs = self.language_decoder.generate(
            inputs_embeds=combined_embeds,
            attention_mask=combined_attention_mask,
            **generation_kwargs
        )

        return outputs


class CustomVLMProcessor:
    """
    Processor for custom VLM (handles image and text preprocessing)
    """

    def __init__(self):
        self.image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        # Set padding token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

    def __call__(self, images=None, text=None, return_tensors="pt", **kwargs):
        """
        Process images and/or text
        """
        encoding = {}

        if images is not None:
            if not isinstance(images, list):
                images = [images]
            image_encoding = self.image_processor(images, return_tensors=return_tensors)
            encoding.update(image_encoding)

        if text is not None:
            # Extract padding and truncation from kwargs if present
            padding = kwargs.pop('padding', True)
            truncation = kwargs.pop('truncation', True)
            max_length = kwargs.pop('max_length', 128)

            text_encoding = self.tokenizer(
                text,
                return_tensors=return_tensors,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                **kwargs
            )
            encoding.update(text_encoding)

        return encoding

    def decode(self, token_ids, skip_special_tokens=True):
        """
        Decode token IDs to text
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def batch_decode(self, token_ids, skip_special_tokens=True):
        """
        Decode batch of token IDs to text
        """
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=skip_special_tokens)


def load_custom_vlm(model_path=None, device='cuda'):
    """
    Load the custom VLM model and processor
    """
    if model_path and os.path.exists(model_path):
        # Load fine-tuned model
        model = VisionLanguageModel()
        state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location=device)
        model.load_state_dict(state_dict)
        model = model.to(device)
        print(f"Loaded custom VLM from {model_path}")
    else:
        # Load pretrained base model
        model = VisionLanguageModel()
        model = model.to(device)
        print("Loaded base custom VLM (not fine-tuned)")

    processor = CustomVLMProcessor()

    model.eval()
    return processor, model


def generate_caption_custom(image, processor, model, device,
                           max_length=50, num_beams=5, temperature=1.0,
                           do_sample=False):
    """
    Generate caption using custom VLM
    """
    # Process image
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs['pixel_values'].to(device)

    # Generation parameters
    gen_kwargs = {
        'max_length': max_length + 32,  # Account for vision tokens
        'num_beams': num_beams if not do_sample else 1,
        'do_sample': do_sample,
        'temperature': temperature if do_sample else 1.0,
        'top_k': 50 if do_sample else None,
        'top_p': 0.95 if do_sample else None,
        'pad_token_id': processor.tokenizer.eos_token_id,
        'eos_token_id': processor.tokenizer.eos_token_id,
    }

    # Generate
    with torch.no_grad():
        output_ids = model.generate(pixel_values=pixel_values, **gen_kwargs)

    # Decode (skip the vision token portion)
    # The first 32 tokens are vision embeddings, actual text starts after
    caption = processor.decode(output_ids[0], skip_special_tokens=True)

    return caption


if __name__ == "__main__":
    # Test the model
    print("Testing Custom VLM Architecture...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Initialize model
    model = VisionLanguageModel()
    model = model.to(device)
    processor = CustomVLMProcessor()

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Test forward pass
    print("\nTesting forward pass...")
    test_image = Image.new('RGB', (224, 224), color='red')
    test_text = "a red square"

    inputs = processor(images=test_image, text=test_text, return_tensors="pt")

    # Move to device
    for key in inputs:
        inputs[key] = inputs[key].to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    print(f"Output logits shape: {outputs.logits.shape}")
    print("Forward pass successful!")

    # Test generation
    print("\nTesting generation...")
    caption = generate_caption_custom(test_image, processor, model, device, max_length=20)
    print(f"Generated caption: {caption}")

    print("\nCustom VLM architecture ready!")
