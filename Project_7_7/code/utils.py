import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from model import get_model

# print detailed  model architecture and statistics
def get_model_summary(model):
    print("=" * 70)
    print("MODEL ARCHITECTURE SUMMARY")
    print("=" * 70)
    model.summary()

    print("\n" + "=" * 70)
    print("MODEL STATISTICS")
    print("=" * 70)

    total_params = model.count_params()
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params

    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Non-Trainable Parameters: {non_trainable_params:,}")
    print(f"Number of Layers: {len(model.layers)}")

# test and display summaries for all three model architectures
def test_all_models():

    models_to_test = [
        ('baseline', 'Baseline CNN'),
        ('improved', 'Improved CNN'),
        ('efficientnet', 'EfficientNetB0 (Transfer Learning)')
    ]

    for model_type, display_name in models_to_test:
        print("\n" + "=" * 70)
        print(f"{display_name.upper()}")

        try:
            # get model (tuple return for transfer learning)
            result = get_model(model_type)
            if isinstance(result, tuple):
                model_instance, _ = result
            else:
                model_instance = result

            # display summary
            get_model_summary(model_instance)
            print(f"\n{display_name} created successfully!\n")

        except Exception as e:
            print(f" Error creating {display_name}: {e}\n")


if __name__ == "__main__":
    test_all_models()