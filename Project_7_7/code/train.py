import os
import pickle
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from dataset import train_generator, val_generator, BATCH_SIZE, IMAGE_SIZE
from model import get_model

MODELS_TO_TRAIN = ['baseline', 'improved', 'efficientnet']

EPOCHS = 20
PATIENCE = 7
LEARNING_RATE = 1e-4
FINE_TUNE_LR = 1e-5

IMG_SIZE = IMAGE_SIZE
NUM_CLASSES = 1

for MODEL_TYPE in MODELS_TO_TRAIN:
    print("\n" + "=" * 50)
    print(f"--- STARTING TRAINING FOR: {MODEL_TYPE.upper()} MODEL ---")
    print("=" * 50)

    SAVE_PATH = f'saved_model/{MODEL_TYPE}_best_model.h5'

    if MODEL_TYPE == 'efficientnet':
        current_lr = FINE_TUNE_LR
    else:
        current_lr = LEARNING_RATE

    checkpoint_callback = ModelCheckpoint(filepath=SAVE_PATH, monitor='val_accuracy', save_best_only=True, mode='max',
                                          verbose=1)
    early_stopping_callback = EarlyStopping(monitor='val_accuracy', patience=PATIENCE, mode='max',
                                            restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-7, verbose=1)

    CALLBACKS = [checkpoint_callback, early_stopping_callback, reduce_lr]

    if MODEL_TYPE == 'efficientnet':
        model, _ = get_model(MODEL_TYPE, img_size=IMG_SIZE, num_classes=NUM_CLASSES)
    else:
        model = get_model(MODEL_TYPE, img_size=IMG_SIZE, num_classes=NUM_CLASSES)

    model.compile(
        optimizer=Adam(learning_rate=current_lr),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    print(f"\n-> Starting training for {MODEL_TYPE}...")

    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=EPOCHS,
        validation_data=val_generator,
        validation_steps=val_generator.samples // val_generator.batch_size,
        callbacks=CALLBACKS
    )

    print("\n-> Training finished.")
    print(f"Best model saved to: {SAVE_PATH}")

    if not os.path.exists('results'):
        os.makedirs('results')

    try:
        with open(f'results/{MODEL_TYPE}_history.pkl', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
        print(f"Training history saved in results/{MODEL_TYPE}_history.pkl")
    except Exception as e:
        print(f"Error saving history file for {MODEL_TYPE}: {e}")

print("\n" + "=" * 50)
print("COMPLETED ALL TRAINING RUNS.")