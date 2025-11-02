# VLM Image Captioning with Fine-tuned BLIP

A Vision-Language Model (VLM) based image captioning application using Salesforce's BLIP (Bootstrapping Language-Image Pre-training) model. This project includes both a pre-trained model and a fine-tuned version, with an interactive Streamlit web interface for generating descriptive captions from images.

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Fine-tuning](#fine-tuning)
- [Project Structure](#project-structure)
- [Results](#results)
- [Technologies Used](#technologies-used)

## Overview

This project implements an image captioning system that generates natural language descriptions of images using vision-language models. The application supports both the pretrained BLIP-Base model and a custom fine-tuned version trained on domain-specific data.

**Key Capabilities:**
- Generate single descriptive captions for uploaded images
- Generate multiple caption variations with different creativity levels
- Interactive web interface built with Streamlit
- Support for both pretrained and fine-tuned models
- Adjustable generation parameters (temperature, beam search, max length)

## Architecture

### BLIP (Bootstrapping Language-Image Pre-training)

**Model:** `Salesforce/blip-image-captioning-base`

BLIP is a state-of-the-art vision-language pre-training framework that achieves excellent performance on various vision-language tasks including image captioning.

#### Architecture Components:

1. **Vision Encoder**
   - Based on Vision Transformer (ViT)
   - Processes input images into visual embeddings
   - Captures spatial and semantic information from images

2. **Text Encoder**
   - Transformer-based encoder
   - Processes text tokens and captions
   - Enables understanding of language context

3. **Text Decoder**
   - Autoregressive decoder
   - Generates captions token-by-token
   - Uses cross-attention to attend to visual features

4. **Multimodal Fusion**
   - Cross-attention mechanism between vision and language
   - Allows the model to ground text generation in visual content
   - Enables contextual understanding of image-text relationships

#### BLIP Training Objectives:

- **Image-Text Contrastive Learning (ITC):** Aligns image and text representations
- **Image-Text Matching (ITM):** Learns fine-grained alignment
- **Language Modeling (LM):** Trains the decoder for caption generation

### Model Specifications

```
Model: BLIP-Base
Parameters: ~223M
Vision Encoder: ViT-B/16
Text Encoder/Decoder: BERT-Base architecture
Input Image Size: 384 x 384 pixels
Vocabulary Size: 30,522 tokens
```

## Features

### Web Application Features:
- **Upload Images:** Support for JPG, JPEG, and PNG formats
- **Sample Images:** Quick testing with pre-loaded sample images
- **Model Selection:** Choose between pretrained and fine-tuned models
- **Advanced Settings:**
  - Max Caption Length (10-50 words)
  - Beam Search Width (1-5 beams)
  - Temperature (0.5-2.0 for creativity control)
- **Multiple Variations:** Generate 3 different caption variations
- **Copy to Clipboard:** Easy caption copying
- **Responsive UI:** Clean, modern interface with custom styling
- **GPU Support:** Automatic GPU detection and usage

### Caption Generation Modes:

1. **Single Caption Mode:**
   - Generates one optimized caption
   - Uses beam search for quality

2. **Multiple Variations Mode:**
   - Generates 3 different captions
   - Increasing temperature for diversity
   - Shows creativity vs accuracy tradeoff

## Installation

### Prerequisites
```bash
Python 3.8+
CUDA-compatible GPU (optional, for faster inference)
```

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/likthvishal/VLM-for-image-captioning.git
cd VLM-for-image-captioning
```

2. **Install dependencies:**
```bash
pip install streamlit torch torchvision transformers pillow
```

3. **Download/Verify Models:**
   - The pretrained model will be automatically downloaded on first run
   - Fine-tuned model should be in `blip_finetuned_best/` directory

## Usage

### Running the Web Application

```bash
streamlit run image_caption_app.py
```

The application will open in your browser at `http://localhost:8501`

### Using the Application:

1. **Upload an Image:**
   - Click "Choose an image..." to upload
   - Or check "Use a sample image" to test with examples

2. **Adjust Settings (Optional):**
   - Expand "Advanced Settings" in the sidebar
   - Adjust max length, beam search, and temperature

3. **Generate Caption:**
   - Click "Generate Caption" button
   - View the generated description

4. **Generate Variations (Optional):**
   - Check "Generate Multiple Captions (3 variations)"
   - See 3 different caption variations

### Python API Usage:

```python
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device)

# Load and process image
image = Image.open("your_image.jpg").convert('RGB')
inputs = processor(image, return_tensors="pt").to(device)

# Generate caption
with torch.no_grad():
    output = model.generate(**inputs, max_length=20, num_beams=3)

caption = processor.decode(output[0], skip_special_tokens=True)
print(caption)
```

## Model Details

### Pretrained Model
- **Model ID:** `Salesforce/blip-image-captioning-base`
- **Training Data:** Large-scale web data (COCO, Visual Genome, etc.)
- **Use Case:** General-purpose image captioning

### Fine-tuned Model
- **Base Model:** BLIP-Base
- **Fine-tuning Dataset:** Flickr8k/Custom dataset
- **Training Epochs:** 70+ epochs with checkpointing
- **Location:** `blip_finetuned_best/`
- **Checkpoints Available:**
  - `checkpoint_epoch_30.pt`
  - `checkpoint_epoch_40.pt`
  - `checkpoint_epoch_50.pt`
  - `checkpoint_epoch_60.pt`
  - `checkpoint_epoch_70.pt`
  - `weights_best.pt` (best validation performance)
  - `weights_final_epoch_70.pt` (final epoch)

### Model Directories:
- `blip_finetuned_best/` - Best performing checkpoint
- `blip_finetuned_epoch_2/` - Early training checkpoint
- `blip_finetuned_epoch_4/` - Early training checkpoint

## Fine-tuning

The model was fine-tuned using the following approach:

### Training Configuration:
```python
Optimizer: AdamW
Learning Rate: 5e-5
Batch Size: 16-32
Epochs: 70
Loss Function: Cross-Entropy
Gradient Clipping: 1.0
```

### Training Process:
1. Started with pretrained BLIP-Base weights
2. Fine-tuned on domain-specific image-caption pairs
3. Used beam search for validation
4. Saved checkpoints every 10 epochs
5. Selected best model based on validation loss/BLEU score

### Dataset:
- Training images in `Images/` directory
- Captions in `captions.txt`
- Multiple captions per image for better generalization

## Project Structure

```
VLM-for-image-captioning/
├── image_caption_app.py              # Main Streamlit application
├── README.md                          # This file
├── captions.txt                       # Training captions
│
├── Images/                            # Training/test images
│   ├── *.jpg                         # Image files
│
├── blip_finetuned_best/              # Best fine-tuned model
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   └── vocab.txt
│
├── blip_finetuned_epoch_2/           # Epoch 2 checkpoint
├── blip_finetuned_epoch_4/           # Epoch 4 checkpoint
│
├── checkpoint_epoch_30.pt            # Training checkpoint
├── checkpoint_epoch_40.pt
├── checkpoint_epoch_50.pt
├── checkpoint_epoch_60.pt
├── checkpoint_epoch_70.pt
│
├── weights_best.pt                   # Best model weights
├── weights_final_epoch_70.pt         # Final epoch weights
├── weights.pt                        # Intermediate weights
│
├── blip_results.png                  # Training results visualization
├── comparison_pretrained_vs_finetuned.png  # Model comparison
├── test_results*.png                 # Various test result visualizations
│
└── *.ipynb                           # Jupyter notebooks for training/testing
```

## Results

### Performance Comparison:

The fine-tuned model shows improved performance on domain-specific images compared to the pretrained model:

- **BLEU Score Improvement:** Fine-tuned model shows higher BLEU scores
- **Domain Specificity:** Better at capturing domain-specific details
- **Caption Quality:** More natural and accurate descriptions

### Visualizations:
- `blip_results.png` - Training loss and metrics over epochs
- `comparison_pretrained_vs_finetuned.png` - Side-by-side model comparison
- `test_results_final.png` - Final test results on validation set

### Example Outputs:

**Pretrained Model:**
- "A dog running in the grass"

**Fine-tuned Model:**
- "A golden retriever running through green grass on a sunny day"

## Technologies Used

### Core Framework:
- **PyTorch** - Deep learning framework
- **Transformers** (Hugging Face) - Model architecture and pretrained weights
- **Streamlit** - Web application framework

### Computer Vision:
- **PIL/Pillow** - Image processing
- **torchvision** - Image transformations

### Model:
- **BLIP (Salesforce)** - Vision-language model
- **Vision Transformer (ViT)** - Visual encoder
- **BERT** - Text encoder/decoder architecture

### Development Tools:
- **Jupyter Notebook** - Experimentation and analysis
- **Git** - Version control

## Use Cases

1. **Accessibility:**
   - Generate alt-text for images for visually impaired users
   - Automatic image descriptions for screen readers

2. **Content Management:**
   - Automatic tagging and cataloging of image libraries
   - Image search and indexing

3. **Social Media:**
   - Automatic caption generation for posts
   - Content recommendation based on image understanding

4. **E-commerce:**
   - Product description generation
   - Image-based search and discovery

5. **Education:**
   - Visual content description for educational materials
   - Accessibility compliance

## Model Performance Metrics

### Quantitative Metrics:
- **BLEU Score:** Measures n-gram overlap with reference captions
- **METEOR:** Considers synonyms and paraphrasing
- **CIDEr:** Consensus-based metric for image description
- **ROUGE-L:** Longest common subsequence based metric

### Qualitative Assessment:
- Caption fluency and grammatical correctness
- Semantic accuracy (objects, actions, attributes)
- Contextual relevance
- Diversity of generated captions

## Future Enhancements

- [ ] Add support for video captioning
- [ ] Implement real-time webcam captioning
- [ ] Multi-language caption generation
- [ ] Integration with larger models (BLIP-2, LLaVA)
- [ ] Fine-tuning UI directly in the app
- [ ] Export captions in various formats (JSON, CSV, XML)
- [ ] Batch processing for multiple images
- [ ] API endpoint for programmatic access

## Acknowledgments

- **Salesforce Research** for the BLIP model
- **Hugging Face** for the Transformers library
- **Flickr** for the image dataset
- **Streamlit** for the web framework

## License

This project is open source and available under the MIT License.

## Contact

For questions, issues, or contributions, please visit the [GitHub repository](https://github.com/likthvishal/VLM-for-image-captioning).

---

**Built with Streamlit | Powered by BLIP | Framework: PyTorch + Transformers**
