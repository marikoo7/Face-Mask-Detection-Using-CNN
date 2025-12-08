import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense,
    Dropout, BatchNormalization, GlobalAveragePooling2D
)
from tensorflow.keras.applications import EfficientNetB0

# baseline CNN model
def build_baseline_cnn(img_size=(128, 128), num_classes=1):
    model = Sequential([
        # block 1: learn basic features (edges, colors)
        Conv2D(32, (3, 3), activation='relu', padding='same',
               input_shape=(img_size[0], img_size[1], 3)),
        MaxPooling2D(pool_size=(2, 2)),

        # block 2: learn more complex patterns
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),

        # block 3: learn high-level features (mask shapes, face coverage)
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),

        # global average pooling instead of flatten (reduces parameters)
        GlobalAveragePooling2D(), # more efficient than flatten

        # dense layers for classification
        Dense(256, activation='relu'),
        Dropout(0.5), # for preventing overfitting

        # output layer using sigmoid for binary classification(mask, no-mask)
        Dense(num_classes, activation='sigmoid')
    ])

    return model

# improved CNN with batch normalization and more layers
def build_improved_cnn(img_size=(128, 128), num_classes=1):
    model = Sequential([
        # block 1
        Conv2D(32, (3, 3), activation='relu', padding='same',
               input_shape=(img_size[0], img_size[1], 3)),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # block 2
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # block 3
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # block 4
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        # global average pooling instead of flatten (reduces parameters)
        GlobalAveragePooling2D(),

        # dense layers
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.3),

        # output layer using sigmoid for binary classification(mask, no-mask)
        Dense(num_classes, activation='sigmoid')
    ])

    return model

# transfer learning with EfficientNetB0
def build_efficientnet_transfer(img_size=(128, 128), num_classes=1, trainable_base=False):
    # load pre-trained EfficientNetB0 without top classification layers
    base_model = EfficientNetB0(
        input_shape=(img_size[0], img_size[1], 3),
        include_top=False,
        weights='imagenet'
    )

    # freeze base model layers, we'll train only the top (can be unfrozen later for fine-tuning)
    base_model.trainable = trainable_base

    # build complete model
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.4),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='sigmoid')
    ])

    return model, base_model

# get any model by name
def get_model(model_type='baseline', img_size=(128, 128), num_classes=1):
    model_type = model_type.lower()

    if model_type == 'baseline':
        return build_baseline_cnn(img_size, num_classes)

    elif model_type == 'improved':
        return build_improved_cnn(img_size, num_classes)

    elif model_type == 'efficientnet':
        return build_efficientnet_transfer(img_size, num_classes)

    else:
        raise ValueError(
            f"Unknown model type: '{model_type}'. "
        )


if __name__ == "__main__":
    # test to verify model builds correctly
    # model summary is in the utils file
    print("Testing baseline model...")
    model = get_model('baseline')
    print(f"Model created successfully")
    print(f"Total parameters: {model.count_params():,}")
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")

    print("\n" + "=" * 70 + "\n")

    print("Testing improved model...")
    model = get_model('improved')
    print(f"Model created successfully")
    print(f"Total parameters: {model.count_params():,}")
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")

    print("\n" + "=" * 70 + "\n")

    print("Testing EfficientNetB0 model...")
    model, base = get_model('efficientnet')
    print(f"Model created successfully")
    print(f"Total parameters: {model.count_params():,}")
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")

