"""
Image Captioning Web App using BLIP
Built with Streamlit
"""

import streamlit as st
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import io
import os

# Page configuration
st.set_page_config(
    page_title="VLM Image Captioning",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem;
        font-size: 16px;
        border-radius: 10px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .caption-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        font-size: 18px;
        text-align: center;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(model_choice):
    """Load BLIP model (cached for performance)"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_choice == "Pretrained BLIP":
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base",
            use_safetensors=True
        ).to(device)
    else:  # Fine-tuned BLIP
        # Try to load from Hugging Face Hub first, then fall back to local path
        model_path_local = "blip_finetuned_best"
        model_path_hf = "high-velo/blip-finetuned-flickr8k"

        try:
            # Try loading from Hugging Face Hub
            processor = BlipProcessor.from_pretrained(model_path_hf)
            model = BlipForConditionalGeneration.from_pretrained(model_path_hf).to(device)
            st.success("Loaded fine-tuned model from Hugging Face Hub")
        except Exception:
            # Fall back to local path
            if os.path.exists(model_path_local):
                processor = BlipProcessor.from_pretrained(model_path_local)
                model = BlipForConditionalGeneration.from_pretrained(model_path_local).to(device)
                st.info("Loaded fine-tuned model from local directory")
            else:
                st.warning("Fine-tuned model not found on Hugging Face or locally. Using pretrained model.")
                processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                model = BlipForConditionalGeneration.from_pretrained(
                    "Salesforce/blip-image-captioning-base",
                    use_safetensors=True
                ).to(device)

    return processor, model, device


def generate_caption(image, processor, model, device, max_length=20, num_beams=1, temperature=1.0, do_sample=False):
    """Generate caption for an image"""

    # Process image
    inputs = processor(image, return_tensors="pt").to(device)

    # Generate caption
    with torch.no_grad():
        if do_sample:
            # Sampling-based generation for diverse outputs
            output = model.generate(
                **inputs,
                max_length=max_length,
                do_sample=True,
                temperature=temperature,
                top_k=50,
                top_p=0.95,
                num_beams=1
            )
        elif num_beams > 1:
            # Beam search for high-quality single output
            output = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams
            )
        else:
            # Greedy decoding
            output = model.generate(
                **inputs,
                max_length=max_length
            )

    # Decode caption
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption


def main():
    # Header
    st.title("VLM Image Captioning")
    st.markdown("### Generate descriptive captions for your images using finetuned Vision-Language-Model.")
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        
        # Model selection
        model_options = ["Pretrained BLIP", "Fine-tuned BLIP"]

        model_choice = st.selectbox(
            "Choose Model",
            model_options,
            help="Select which model to use for captioning"
        )
        
        st.markdown("---")
        
        # Advanced settings
        with st.expander("Advanced Settings"):
            max_length = st.slider(
                "Max Caption Length",
                min_value=10,
                max_value=50,
                value=20,
                help="Maximum number of words in caption"
            )
            
            num_beams = st.slider(
                "Beam Search Width",
                min_value=1,
                max_value=5,
                value=3,
                help="Higher = more diverse but slower"
            )
            
            temperature = st.slider(
                "Temperature",
                min_value=0.5,
                max_value=2.0,
                value=1.0,
                step=0.1,
                help="Higher = more creative but less accurate"
            )
        
        st.markdown("---")
        
        # Info
        st.info("""
        **How to use:**
        1. Upload an image
        2. Click "Generate Caption"
        3. Get AI-generated description!
        """)
        
        # Model info
        st.markdown("---")
        st.markdown("**Model Info:**")
        device_name = "GPU" if torch.cuda.is_available() else "CPU"
        st.text(f"Device: {device_name}")
        st.text(f"Model: BLIP-Base")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload JPG, JPEG, or PNG images"
        )

        # Sample images option (only show if Images directory exists)
        sample_dir = os.path.join(os.path.dirname(__file__), "Images")

        if os.path.exists(sample_dir) and os.path.isdir(sample_dir):
            sample_images = [f for f in os.listdir(sample_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

            if sample_images:
                use_sample = st.checkbox("Or use a sample image", value=False)

                if use_sample:
                    import random
                    # Show a random selection of up to 10 sample images
                    display_samples = random.sample(sample_images, min(10, len(sample_images)))
                    selected_sample = st.selectbox("Select sample image", display_samples)
                    sample_path = os.path.join(sample_dir, selected_sample)
                    uploaded_file = open(sample_path, 'rb')
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Image info
            st.text(f"Image size: {image.size[0]} x {image.size[1]} pixels")
    
    with col2:
        st.subheader("Generated Caption")

        if uploaded_file is not None:
            # Initialize session state
            if 'caption_generated' not in st.session_state:
                st.session_state.caption_generated = False
            if 'caption' not in st.session_state:
                st.session_state.caption = ""

            # Generate button
            if st.button("Generate Caption", use_container_width=True):
                st.session_state.caption_generated = True

            # Generate caption if button was clicked
            if st.session_state.caption_generated:
                with st.spinner("AI is analyzing the image..."):
                    try:
                        # Load model
                        processor, model, device = load_model(model_choice)

                        # Generate caption
                        caption = generate_caption(
                            image,
                            processor,
                            model,
                            device,
                            max_length=max_length,
                            num_beams=num_beams,
                            temperature=temperature
                        )

                        # Store caption in session state
                        st.session_state.caption = caption

                        # Display result
                        st.success("Caption generated successfully!")

                        # Caption box
                        st.markdown(f"""
                        <div class="caption-box">
                            <h3>Caption:</h3>
                            <p style="font-size: 22px; font-weight: bold; color: #1976d2;">
                                {caption}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                        # Additional options
                        st.markdown("---")

                        col_a, col_b = st.columns(2)

                        with col_a:
                            if st.button("Generate Another", use_container_width=True):
                                st.session_state.caption_generated = False
                                st.session_state.caption = ""
                                st.rerun()

                        with col_b:
                            # Copy to clipboard (text)
                            st.text_input("Copy caption:", value=caption, key="caption_copy")

                        # Generate multiple captions
                        if st.checkbox("Generate Multiple Captions (3 variations)"):
                            with st.spinner("Generating variations..."):
                                st.markdown("### Caption Variations:")

                                # Generate diverse captions using sampling
                                temperatures = [0.7, 1.0, 1.3]  # Different temperatures for diversity
                                variations = []

                                for temp in temperatures:
                                    varied_caption = generate_caption(
                                        image,
                                        processor,
                                        model,
                                        device,
                                        max_length=max_length,
                                        temperature=temp,
                                        do_sample=True  # Enable sampling for diversity
                                    )

                                    # Only add if it's different from previous captions
                                    if varied_caption not in variations:
                                        variations.append(varied_caption)
                                        st.markdown(f"**{len(variations)}.** {varied_caption}")

                                # If we didn't get enough unique variations, generate more
                                attempts = 0
                                while len(variations) < 3 and attempts < 5:
                                    temp_variation = generate_caption(
                                        image,
                                        processor,
                                        model,
                                        device,
                                        max_length=max_length,
                                        temperature=1.5,
                                        do_sample=True
                                    )
                                    if temp_variation not in variations:
                                        variations.append(temp_variation)
                                        st.markdown(f"**{len(variations)}.** {temp_variation}")
                                    attempts += 1

                    except Exception as e:
                        st.error(f"Error generating caption: {str(e)}")
                        st.info("Try uploading a different image or check model settings.")
        else:
            st.info("Upload an image to get started!")
            
            # Example section
            st.markdown("---")
            st.markdown("### Example Use Cases:")
            st.markdown("""
            - Social media post descriptions
            - Accessibility for visually impaired
            - Automatic image organization
            - Image search and indexing
            - Content management systems
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>Built with Streamlit</p>
        <p>Model: BLIP-Base | Framework: PyTorch + Transformers</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
