import os
import shutil
from pathlib import Path
from PIL import Image, ImageOps
import numpy as np
import random
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input

DATA_DIR = "raw_images_combined"
OUTPUT_DIR = "dataset_split"
IMAGE_SIZE = (128, 128)
CLASSES = ["with_mask", "without_mask"]
BATCH_SIZE = 32

# if os.path.exists(OUTPUT_DIR):
#     shutil.rmtree(OUTPUT_DIR)

# for split in ["train", "val", "test"]:
#     for cls in CLASSES:
#         os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)

# images = []
# labels = []

# for cls in CLASSES:
#     folder = os.path.join(DATA_DIR, cls)
#     for img_name in os.listdir(folder):
#         img_path = os.path.join(folder, img_name)
#         if os.path.isfile(img_path):
#             images.append(img_path)
#             labels.append(cls)

# train_imgs, temp_imgs, train_lbls, temp_lbls = train_test_split(
#     images, labels, test_size=0.30, stratify=labels, random_state=42
# )
# val_imgs, test_imgs, val_lbls, test_lbls = train_test_split(
#     temp_imgs, temp_lbls, test_size=0.50, stratify=temp_lbls, random_state=42
# )

# def is_corrupted(image_path):
#     try:
#         img = Image.open(image_path)
#         img.verify()
#         return False
#     except:
#         return True

# def process_and_copy(imgs, lbls, split_name):
#     print(f"Starting {split_name} data copy to {IMAGE_SIZE}...")
#     for img_path, lbl in zip(imgs, lbls):
#         if is_corrupted(img_path):
#             print(f" Corrupted image skipped: {img_path}")
#             continue
#         img = Image.open(img_path).convert("RGB")
#         img = img.resize(IMAGE_SIZE)
#         filename = os.path.basename(img_path)
#         dst = os.path.join(OUTPUT_DIR, split_name, lbl, filename)
#         img.save(dst)

# process_and_copy(train_imgs, train_lbls, "train")
# process_and_copy(val_imgs, val_lbls, "val")
# process_and_copy(test_imgs, test_lbls, "test")

# for split in ["train","val","test"]:
#     print(f"--- {split} ---")
#     for cls in CLASSES:
#         path = os.path.join(OUTPUT_DIR, split, cls)
#         count = len(os.listdir(path))
#         print(f"{cls}: {count} images")


def get_generators(model_type='baseline'):

    print(f"Creating generators for {model_type} model...")

    # EfficientNet needs special preprocessing
    if model_type.lower() == 'efficientnet':

        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            shear_range=0.1,
            horizontal_flip=True,
        )

        test_val_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input
        )

    else:

        train_datagen = ImageDataGenerator(
            rescale=1. / 255,  # ← CRITICAL: Regular rescaling!
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            shear_range=0.1,
            horizontal_flip=True,
            brightness_range=(0.8, 1.2),
        )

        test_val_datagen = ImageDataGenerator(
            rescale=1. / 255  # ← CRITICAL: Regular rescaling!
        )

    # Create generators
    train_gen = train_datagen.flow_from_directory(
        os.path.join(OUTPUT_DIR, "train"),
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=True
    )

    val_gen = test_val_datagen.flow_from_directory(
        os.path.join(OUTPUT_DIR, "val"),
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )

    test_gen = test_val_datagen.flow_from_directory(
        os.path.join(OUTPUT_DIR, "test"),
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )

    print(f"   Generators created successfully")
    print(f"   Train: {train_gen.samples} samples")
    print(f"   Val: {val_gen.samples} samples")
    print(f"   Test: {test_gen.samples} samples")

    return train_gen, val_gen, test_gen



print("Creating default generators...")

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    shear_range=0.1,
    horizontal_flip=True,
)

test_val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_generator = train_datagen.flow_from_directory(
    os.path.join(OUTPUT_DIR, "train"),
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

val_generator = test_val_datagen.flow_from_directory(
    os.path.join(OUTPUT_DIR, "val"),
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

test_generator = test_val_datagen.flow_from_directory(
    os.path.join(OUTPUT_DIR, "test"),
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

print("train samples:", train_generator.samples)
print("val samples:", val_generator.samples)
print("test samples:", test_generator.samples)
print("classes:", train_generator.class_indices)
