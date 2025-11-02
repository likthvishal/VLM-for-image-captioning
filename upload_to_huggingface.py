"""
Script to upload fine-tuned BLIP model to Hugging Face Hub
"""

import os
from huggingface_hub import HfApi, login, create_repo
import sys

def upload_model_to_hf():
    """Upload the fine-tuned BLIP model to Hugging Face Hub"""

    # Configuration
    model_path = "blip_finetuned_best"
    repo_name = "blip-finetuned-flickr8k"  # You can change this name

    print("=" * 70)
    print("BLIP Fine-tuned Model Upload to Hugging Face Hub")
    print("=" * 70)
    print()

    # Check if model directory exists
    if not os.path.exists(model_path):
        print(f"❌ Error: Model directory '{model_path}' not found!")
        print(f"   Make sure your fine-tuned model is in the current directory.")
        return False

    # List model files
    model_files = os.listdir(model_path)
    print(f"📁 Found model files in '{model_path}':")
    for file in model_files[:10]:  # Show first 10 files
        print(f"   - {file}")
    if len(model_files) > 10:
        print(f"   ... and {len(model_files) - 10} more files")
    print()

    # Get Hugging Face token
    print("🔑 Hugging Face Authentication")
    print("-" * 70)
    print("You need a Hugging Face account and access token:")
    print("1. Go to https://huggingface.co/settings/tokens")
    print("2. Create a new token with 'write' access")
    print("3. Copy the token and paste it below")
    print()

    token = input("Enter your Hugging Face token (or press Enter if already logged in): ").strip()

    try:
        # Login to Hugging Face
        if token:
            login(token=token, add_to_git_credential=True)
            print("✅ Successfully logged in to Hugging Face!")
        else:
            print("ℹ️  Using existing Hugging Face credentials...")
        print()

        # Get username
        api = HfApi()
        user_info = api.whoami()
        username = user_info['name']

        print(f"👤 Logged in as: {username}")
        print()

        # Create repository ID
        repo_id = f"{username}/{repo_name}"
        print(f"📦 Repository: {repo_id}")
        print()

        # Confirm upload
        confirm = input("Do you want to proceed with the upload? (yes/no): ").strip().lower()
        if confirm != 'yes':
            print("❌ Upload cancelled.")
            return False

        print()
        print("🚀 Starting upload...")
        print("-" * 70)

        # Create repository (if it doesn't exist)
        try:
            create_repo(repo_id, exist_ok=True, repo_type="model")
            print(f"✅ Repository '{repo_id}' is ready")
        except Exception as e:
            print(f"ℹ️  Repository might already exist: {e}")

        # Upload model files
        print(f"📤 Uploading model files from '{model_path}'...")
        api.upload_folder(
            folder_path=model_path,
            repo_id=repo_id,
            repo_type="model",
            commit_message="Upload fine-tuned BLIP model trained on Flickr8k (70+ epochs)"
        )

        print()
        print("=" * 70)
        print("✅ SUCCESS! Model uploaded to Hugging Face Hub")
        print("=" * 70)
        print()
        print(f"🌐 Model URL: https://huggingface.co/{repo_id}")
        print()
        print("Next steps:")
        print(f"1. Visit https://huggingface.co/{repo_id} to view your model")
        print("2. Update image_caption_app.py to use this model")
        print(f"   - Change model_path to: '{repo_id}'")
        print("3. Push changes to GitHub to deploy on Streamlit Cloud")
        print()

        # Save repo_id to a file for easy reference
        with open("huggingface_model_id.txt", "w") as f:
            f.write(repo_id)
        print(f"💾 Model ID saved to 'huggingface_model_id.txt'")
        print()

        return True

    except Exception as e:
        print()
        print("=" * 70)
        print("❌ ERROR during upload")
        print("=" * 70)
        print(f"Error: {str(e)}")
        print()
        print("Common issues:")
        print("1. Invalid token - make sure you copied the full token")
        print("2. Token doesn't have 'write' permissions")
        print("3. Network issues - check your internet connection")
        print()
        return False

if __name__ == "__main__":
    print()
    success = upload_model_to_hf()
    print()

    if success:
        sys.exit(0)
    else:
        sys.exit(1)
