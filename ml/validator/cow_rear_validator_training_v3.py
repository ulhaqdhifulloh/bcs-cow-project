"""
Cow Rear Validator Training Script (V3 - Using Real Datasets)
==============================================================
Script untuk melatih model binary classification yang memvalidasi
apakah gambar adalah bagian belakang sapi atau bukan.

IMPROVEMENTS:
- Uses real cow datasets with labeled back view and side view
- Combines multiple data sources for better generalization
- Dataset: Cattle side view and back view (Mendeley Data)
- Dataset: Cow Pose Estimation (Kaggle)

Kelas:
- cow_rear: Gambar bagian belakang sapi
- not_cow_rear: Gambar bukan bagian belakang sapi (side view + other cow poses)

Usage:
    python cow_rear_validator_training_v3.py
"""

import os
import shutil
import random
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
import yaml

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
BASE_DIR = Path(__file__).parent.parent  # ml folder
VALIDATOR_DIR = BASE_DIR / "validator"
BCS_DATASET_DIR = BASE_DIR / "data" / "processed"

# External datasets
CATTLE_DATASET_DIR = VALIDATOR_DIR / "dataset" / "Cattle side view and back view dataset"
COW_POSE_DATASET_DIR = VALIDATOR_DIR / "dataset" / "Cow Pose Estimation Dataset"

# Output
NEW_DATASET_DIR = VALIDATOR_DIR / "dataset_v3"
MODEL_OUTPUT_DIR = BASE_DIR / "models" / "cow_rear_validator"

# Training settings
MODEL_BASE = "yolov8n-cls"
EPOCHS = 100  # More epochs for better learning
IMAGE_SIZE = 224
BATCH_SIZE = 16
TRAIN_RATIO = 0.8

# Sample settings
NUM_BCS_SAMPLES = 200  # Sample from your BCS dataset for cow_rear

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def resize_and_save(src_path, dst_path, size=(224, 224)):
    """Resize image and save to destination."""
    try:
        img = Image.open(src_path)
        img = img.convert('RGB')
        img = img.resize(size, Image.LANCZOS)
        img.save(dst_path, 'JPEG', quality=95)
        return True
    except Exception as e:
        print(f"   ⚠ Error processing {src_path}: {e}")
        return False


# ============================================================================
# STEP 1: PREPARE DATASET
# ============================================================================

def create_directory_structure():
    """Membuat struktur folder untuk dataset baru."""
    print("\n📁 Creating directory structure...")
    
    # Clear existing dataset
    if NEW_DATASET_DIR.exists():
        print("   🗑️ Clearing existing dataset...")
        shutil.rmtree(NEW_DATASET_DIR)
    
    dirs_to_create = [
        NEW_DATASET_DIR / "train" / "cow_rear",
        NEW_DATASET_DIR / "train" / "not_cow_rear",
        NEW_DATASET_DIR / "val" / "cow_rear",
        NEW_DATASET_DIR / "val" / "not_cow_rear",
    ]
    
    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"   ✓ Created: {dir_path}")
    
    print("   Done!")


def prepare_cow_rear_from_cattle_dataset():
    """Prepare cow_rear images from Cattle side view and back view dataset."""
    print("\n🐄 Processing BACK VIEW images from Cattle dataset...")
    
    back_view_dir = CATTLE_DATASET_DIR / "back view"
    
    if not back_view_dir.exists():
        print(f"   ⚠ Directory not found: {back_view_dir}")
        return 0, 0
    
    all_images = list(back_view_dir.glob("*.png")) + list(back_view_dir.glob("*.jpg"))
    random.shuffle(all_images)
    
    print(f"   Found {len(all_images)} back view images")
    
    # Split train/val
    split_idx = int(len(all_images) * TRAIN_RATIO)
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]
    
    # Process and copy
    train_count = 0
    for i, img_path in enumerate(train_images):
        dst = NEW_DATASET_DIR / "train" / "cow_rear" / f"cattle_back_{i:04d}.jpg"
        if resize_and_save(img_path, dst):
            train_count += 1
    
    val_count = 0
    for i, img_path in enumerate(val_images):
        dst = NEW_DATASET_DIR / "val" / "cow_rear" / f"cattle_back_{i:04d}.jpg"
        if resize_and_save(img_path, dst):
            val_count += 1
    
    print(f"   ✓ Processed {train_count} train + {val_count} val images")
    return train_count, val_count


def prepare_cow_rear_from_bcs():
    """Add cow_rear images from BCS dataset for variety."""
    print(f"\n🐄 Sampling {NUM_BCS_SAMPLES} images from BCS dataset...")
    
    bcs_train_dir = BCS_DATASET_DIR / "train"
    
    if not bcs_train_dir.exists():
        print(f"   ⚠ BCS dataset not found: {bcs_train_dir}")
        return 0, 0
    
    all_images = []
    for class_folder in bcs_train_dir.iterdir():
        if class_folder.is_dir():
            for img_file in class_folder.iterdir():
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']:
                    all_images.append(img_file)
    
    print(f"   Found {len(all_images)} images in BCS dataset")
    
    # Sample
    sampled = random.sample(all_images, min(NUM_BCS_SAMPLES, len(all_images)))
    random.shuffle(sampled)
    
    # Split train/val
    split_idx = int(len(sampled) * TRAIN_RATIO)
    train_images = sampled[:split_idx]
    val_images = sampled[split_idx:]
    
    # Get existing count for naming
    existing_train = len(list((NEW_DATASET_DIR / "train" / "cow_rear").glob("*")))
    existing_val = len(list((NEW_DATASET_DIR / "val" / "cow_rear").glob("*")))
    
    # Copy
    train_count = 0
    for i, img_path in enumerate(train_images):
        dst = NEW_DATASET_DIR / "train" / "cow_rear" / f"bcs_{existing_train + i:04d}.jpg"
        if resize_and_save(img_path, dst):
            train_count += 1
    
    val_count = 0
    for i, img_path in enumerate(val_images):
        dst = NEW_DATASET_DIR / "val" / "cow_rear" / f"bcs_{existing_val + i:04d}.jpg"
        if resize_and_save(img_path, dst):
            val_count += 1
    
    print(f"   ✓ Added {train_count} train + {val_count} val images")
    return train_count, val_count


def prepare_not_cow_rear_from_cattle_dataset():
    """Prepare not_cow_rear images from Cattle dataset (side view)."""
    print("\n🐄 Processing SIDE VIEW images from Cattle dataset...")
    
    side_view_dir = CATTLE_DATASET_DIR / "side view"
    
    if not side_view_dir.exists():
        print(f"   ⚠ Directory not found: {side_view_dir}")
        return 0, 0
    
    all_images = list(side_view_dir.glob("*.png")) + list(side_view_dir.glob("*.jpg"))
    random.shuffle(all_images)
    
    print(f"   Found {len(all_images)} side view images")
    
    # Split train/val
    split_idx = int(len(all_images) * TRAIN_RATIO)
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]
    
    # Process and copy
    train_count = 0
    for i, img_path in enumerate(train_images):
        dst = NEW_DATASET_DIR / "train" / "not_cow_rear" / f"cattle_side_{i:04d}.jpg"
        if resize_and_save(img_path, dst):
            train_count += 1
    
    val_count = 0
    for i, img_path in enumerate(val_images):
        dst = NEW_DATASET_DIR / "val" / "not_cow_rear" / f"cattle_side_{i:04d}.jpg"
        if resize_and_save(img_path, dst):
            val_count += 1
    
    print(f"   ✓ Processed {train_count} train + {val_count} val images")
    return train_count, val_count


def prepare_not_cow_rear_from_pose_dataset():
    """Add images from Cow Pose dataset (mostly side view)."""
    print("\n🐄 Processing images from Cow Pose dataset...")
    
    # Collect images from train and val folders
    all_images = []
    
    train_dir = COW_POSE_DATASET_DIR / "images" / "train"
    val_dir = COW_POSE_DATASET_DIR / "images" / "val"
    
    for dir_path in [train_dir, val_dir]:
        if dir_path.exists():
            all_images.extend(list(dir_path.glob("*.jpg")))
            all_images.extend(list(dir_path.glob("*.png")))
    
    print(f"   Found {len(all_images)} images from Cow Pose dataset")
    
    random.shuffle(all_images)
    
    # Split train/val
    split_idx = int(len(all_images) * TRAIN_RATIO)
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]
    
    # Get existing count
    existing_train = len(list((NEW_DATASET_DIR / "train" / "not_cow_rear").glob("*")))
    existing_val = len(list((NEW_DATASET_DIR / "val" / "not_cow_rear").glob("*")))
    
    # Process and copy
    train_count = 0
    for i, img_path in enumerate(train_images):
        dst = NEW_DATASET_DIR / "train" / "not_cow_rear" / f"pose_{existing_train + i:04d}.jpg"
        if resize_and_save(img_path, dst):
            train_count += 1
    
    val_count = 0
    for i, img_path in enumerate(val_images):
        dst = NEW_DATASET_DIR / "val" / "not_cow_rear" / f"pose_{existing_val + i:04d}.jpg"
        if resize_and_save(img_path, dst):
            val_count += 1
    
    print(f"   ✓ Added {train_count} train + {val_count} val images")
    return train_count, val_count


def create_data_yaml():
    """Membuat file data.yaml untuk training YOLO."""
    print("\n📝 Creating data.yaml...")
    
    data_config = {
        'path': str(NEW_DATASET_DIR.absolute()),
        'train': 'train',
        'val': 'val',
        'nc': 2,
        'names': {
            0: 'cow_rear',
            1: 'not_cow_rear'
        }
    }
    
    yaml_path = NEW_DATASET_DIR / "data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)
    
    print(f"   ✓ Created: {yaml_path}")
    
    # Print dataset summary
    train_cow = len(list((NEW_DATASET_DIR / "train" / "cow_rear").glob("*")))
    train_not = len(list((NEW_DATASET_DIR / "train" / "not_cow_rear").glob("*")))
    val_cow = len(list((NEW_DATASET_DIR / "val" / "cow_rear").glob("*")))
    val_not = len(list((NEW_DATASET_DIR / "val" / "not_cow_rear").glob("*")))
    
    print(f"\n   📊 Dataset Summary:")
    print(f"   ┌─────────────────┬───────────┬───────────┐")
    print(f"   │ Class           │ Train     │ Val       │")
    print(f"   ├─────────────────┼───────────┼───────────┤")
    print(f"   │ cow_rear        │ {train_cow:>9} │ {val_cow:>9} │")
    print(f"   │ not_cow_rear    │ {train_not:>9} │ {val_not:>9} │")
    print(f"   ├─────────────────┼───────────┼───────────┤")
    print(f"   │ Total           │ {train_cow+train_not:>9} │ {val_cow+val_not:>9} │")
    print(f"   └─────────────────┴───────────┴───────────┘")
    
    return yaml_path


# ============================================================================
# STEP 2: TRAIN MODEL
# ============================================================================

def train_model():
    """Melatih model YOLOv8 classification."""
    print(f"\n🚀 Training model...")
    print(f"   Base model: {MODEL_BASE}")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Image size: {IMAGE_SIZE}")
    print(f"   Batch size: {BATCH_SIZE}")
    
    # Clear old model if exists
    if MODEL_OUTPUT_DIR.exists():
        shutil.rmtree(MODEL_OUTPUT_DIR)
    
    # Load base model
    model = YOLO(MODEL_BASE)
    
    # Train with data augmentation
    results = model.train(
        data=str(NEW_DATASET_DIR),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        project=str(MODEL_OUTPUT_DIR.parent),
        name=MODEL_OUTPUT_DIR.name,
        exist_ok=True,
        verbose=True,
        patience=20,  # Early stopping
        # Data augmentation for better generalization
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15,
        translate=0.1,
        scale=0.5,
        flipud=0.3,
        fliplr=0.5,
        mosaic=0.5,
    )
    
    return results


# ============================================================================
# STEP 3: TEST MODEL
# ============================================================================

def test_model():
    """Test model dengan beberapa gambar sample."""
    print("\n🧪 Testing model...")
    
    model_path = MODEL_OUTPUT_DIR / "weights" / "best.pt"
    
    if not model_path.exists():
        print(f"   ⚠ Model not found at {model_path}")
        return
    
    model = YOLO(model_path)
    
    # Test dengan gambar cow_rear
    cow_rear_test_dir = NEW_DATASET_DIR / "val" / "cow_rear"
    test_images = list(cow_rear_test_dir.glob("*"))[:10]
    
    print("\n   Testing with COW_REAR images (should predict cow_rear):")
    correct_cow = 0
    for img_path in test_images:
        results = model(img_path, verbose=False)
        pred_class = results[0].probs.top1
        pred_conf = results[0].probs.top1conf.item()
        class_name = results[0].names[pred_class]
        status = "✓" if class_name == "cow_rear" else "✗"
        if class_name == "cow_rear":
            correct_cow += 1
        print(f"   {status} {img_path.name}: {class_name} ({pred_conf:.2%})")
    
    # Test dengan gambar not_cow_rear
    not_cow_rear_test_dir = NEW_DATASET_DIR / "val" / "not_cow_rear"
    test_images = list(not_cow_rear_test_dir.glob("*"))[:10]
    
    print("\n   Testing with NOT_COW_REAR images (should predict not_cow_rear):")
    correct_not = 0
    for img_path in test_images:
        results = model(img_path, verbose=False)
        pred_class = results[0].probs.top1
        pred_conf = results[0].probs.top1conf.item()
        class_name = results[0].names[pred_class]
        status = "✓" if class_name == "not_cow_rear" else "✗"
        if class_name == "not_cow_rear":
            correct_not += 1
        print(f"   {status} {img_path.name}: {class_name} ({pred_conf:.2%})")
    
    total_correct = correct_cow + correct_not
    print(f"\n   📊 Test Accuracy: {total_correct}/20 ({total_correct/20:.1%})")
    print(f"       - cow_rear: {correct_cow}/10")
    print(f"       - not_cow_rear: {correct_not}/10")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("🐄 COW REAR VALIDATOR - Training Script V3")
    print("=" * 60)
    print("\nUsing real labeled datasets:")
    print("  - Cattle side view and back view dataset (Mendeley)")
    print("  - Cow Pose Estimation Dataset (Kaggle)")
    print("  - Your BCS dataset (for variety)")
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Step 1: Prepare dataset
    print("\n" + "=" * 60)
    print("STEP 1: PREPARE DATASET")
    print("=" * 60)
    
    create_directory_structure()
    
    # Cow rear class
    prepare_cow_rear_from_cattle_dataset()
    prepare_cow_rear_from_bcs()
    
    # Not cow rear class
    prepare_not_cow_rear_from_cattle_dataset()
    prepare_not_cow_rear_from_pose_dataset()
    
    create_data_yaml()
    
    # Step 2: Train model
    print("\n" + "=" * 60)
    print("STEP 2: TRAIN MODEL")
    print("=" * 60)
    
    train_model()
    
    # Step 3: Test model
    print("\n" + "=" * 60)
    print("STEP 3: TEST MODEL")
    print("=" * 60)
    
    test_model()
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nModel saved at: {MODEL_OUTPUT_DIR / 'weights' / 'best.pt'}")
    print("\nNext steps:")
    print("1. Restart the API: python main.py")
    print("2. Test with cow rear images from internet")


if __name__ == "__main__":
    main()
