import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import shutil
from pathlib import Path
from PIL import Image, ImageOps
import numpy as np
import random
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator


DATA_DIR = "raw_images_combined"
OUTPUT_DIR = "dataset_split"
IMAGE_SIZE = (128, 128)
CLASSES = ["with_mask", "without_mask"]
BATCH_SIZE = 32


for split in ["train", "val", "test"]:
    for cls in CLASSES:
        os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)


images = []
labels = []

for cls in CLASSES:
    folder = os.path.join(DATA_DIR, cls)
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        if os.path.isfile(img_path):
            images.append(img_path)
            labels.append(cls)


train_imgs, temp_imgs, train_lbls, temp_lbls = train_test_split(
    images, labels, test_size=0.30, stratify=labels, random_state=42
)
val_imgs, test_imgs, val_lbls, test_lbls = train_test_split(
    temp_imgs, temp_lbls, test_size=0.50, stratify=temp_lbls, random_state=42
)


def is_corrupted(image_path):
    try:
        img = Image.open(image_path)
        img.verify()
        return False
    except:
        return True

def process_and_copy(imgs, lbls, split_name):
    for img_path, lbl in zip(imgs, lbls):
        if is_corrupted(img_path):
            print(f"⚠️ Corrupted image skipped: {img_path}")
            continue
        img = Image.open(img_path).convert("RGB")
        img = img.resize(IMAGE_SIZE)
        filename = os.path.basename(img_path)
        dst = os.path.join(OUTPUT_DIR, split_name, lbl, filename)
        img.save(dst)

process_and_copy(train_imgs, train_lbls, "train")
process_and_copy(val_imgs, val_lbls, "val")
process_and_copy(test_imgs, test_lbls, "test")

print("✔️ Dataset split and basic preprocessing done!")


for split in ["train","val","test"]:
    print(f"--- {split} ---")
    for cls in CLASSES:
        path = os.path.join(OUTPUT_DIR, split, cls)
        count = len(os.listdir(path))
        print(f"{cls}: {count} images")

def add_gaussian_noise(img):
    arr = np.array(img) / 255.0
    noise = np.random.normal(0, 0.05, arr.shape)
    arr = np.clip(arr + noise, 0, 1)
    img = Image.fromarray((arr * 255).astype(np.uint8))
    return img

def custom_augmentation(img):
    img = Image.fromarray(img.astype('uint8'), 'RGB')

    img = add_gaussian_noise(img)

    if random.random() < 0.5:
        coeffs = [random.uniform(-0.1,0.1) for _ in range(8)]
        img = img.transform(img.size, Image.PERSPECTIVE, coeffs)

    if random.random() < 0.5:
        crop_size = random.randint(int(0.8*IMAGE_SIZE[0]), IMAGE_SIZE[0])
        img = ImageOps.fit(img, (crop_size, crop_size))
        img = img.resize(IMAGE_SIZE)
    return np.array(img)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    shear_range=0.1,
    horizontal_flip=True,
    brightness_range=(0.8,1.2),
    preprocessing_function=custom_augmentation,
    fill_mode='nearest'
)

test_val_datagen = ImageDataGenerator(rescale=1./255)

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

print("✔️ Advanced Data Generators ready — Train/Val/Test loaded with all augmentations!")
