"""
evaluate.py
Evaluation and visualization script for Face Mask Detection
Team Member: Person 4
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)

from dataset import get_generators


# ============================================================
# CONFIGURATION
# ============================================================
MODELS_TO_EVALUATE = ['baseline','improved','efficientnet'] #, ',
SAVED_MODEL_DIR = '../saved_model'
RESULTS_DIR = '../results'

os.makedirs(RESULTS_DIR, exist_ok=True)

print("="*70)
print("FACE MASK DETECTION - MODEL EVALUATION")
print("="*70)


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def plot_training_history(model_type):
    """Plot training curves"""
    history_path = f'{RESULTS_DIR}/{model_type}_history.pkl'

    if not os.path.exists(history_path):
        print(f"⚠ History file not found: {history_path}")
        return

    with open(history_path, 'rb') as f:
        history = pickle.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    epochs = range(1, len(history['accuracy']) + 1)

    # Accuracy
    ax1.plot(epochs, history['accuracy'], 'b-o', label='Training', linewidth=2, markersize=4)
    ax1.plot(epochs, history['val_accuracy'], 'r-o', label='Validation', linewidth=2, markersize=4)
    ax1.set_title(f'{model_type.upper()} - Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])

    # Loss
    ax2.plot(epochs, history['loss'], 'b-o', label='Training', linewidth=2, markersize=4)
    ax2.plot(epochs, history['val_loss'], 'r-o', label='Validation', linewidth=2, markersize=4)
    ax2.set_title(f'{model_type.upper()} - Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = f'{RESULTS_DIR}/{model_type}_accuracy_loss_curves.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {save_path}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, model_type, class_names):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                annot_kws={'size': 16, 'fontweight': 'bold'})

    plt.title(f'{model_type.upper()} - Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=13)
    plt.xlabel('Predicted Label', fontsize=13)

    plt.tight_layout()
    save_path = f'{RESULTS_DIR}/{model_type}_confusion_matrix.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {save_path}")
    plt.close()


def plot_roc_curve(y_true, y_pred_proba, model_type):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=3,
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=13, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=13, fontweight='bold')
    plt.title(f'{model_type.upper()} - ROC Curve', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = f'{RESULTS_DIR}/{model_type}_roc_curve.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {save_path}")
    plt.close()

    return roc_auc


def plot_sample_predictions(model, test_generator, model_type, num_samples=12):
    """Plot sample predictions"""
    test_generator.reset()

    images, labels = next(test_generator)
    predictions = model.predict(images[:num_samples], verbose=0)

    class_names = list(test_generator.class_indices.keys())

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.ravel()

    for i in range(num_samples):
        # Get image - handle preprocessing
        img = images[i].copy()

        # Reverse EfficientNet preprocessing if needed
        if model_type == 'efficientnet':
            # EfficientNet uses preprocessing_function, so values might be scaled differently
            # Simple approach: just clip to [0,1]
            img = np.clip(img, 0, 1)
        else:
            # For baseline/improved, already in [0,1] from rescale
            img = np.clip(img, 0, 1)

        axes[i].imshow(img)
        axes[i].axis('off')

        # Prediction
        pred_proba = predictions[i][0]
        pred_class_idx = 1 if pred_proba > 0.5 else 0
        pred_class = class_names[pred_class_idx]
        confidence = pred_proba if pred_proba > 0.5 else 1 - pred_proba

        # True label
        true_class_idx = int(labels[i])
        true_class = class_names[true_class_idx]

        color = 'green' if pred_class == true_class else 'red'

        axes[i].set_title(
            f'True: {true_class}\nPred: {pred_class}\nConf: {confidence*100:.1f}%',
            color=color, fontweight='bold', fontsize=10
        )

    plt.suptitle(f'{model_type.upper()} - Sample Predictions', fontsize=16, fontweight='bold')
    plt.tight_layout()

    save_path = f'{RESULTS_DIR}/{model_type}_sample_predictions.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {save_path}")
    plt.close()


def evaluate_model(model_type):
    """Complete evaluation for one model"""
    print("\n" + "="*70)
    print(f"EVALUATING: {model_type.upper()} MODEL")
    print("="*70)

    model_path = f'{SAVED_MODEL_DIR}/{model_type}_best_model.h5'

    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return None

    print(f"\n📂 Loading model...")
    model = load_model(model_path)

    print("📂 Loading test data...")
    _, _, test_generator = get_generators(model_type)

    print(f"   Test samples: {test_generator.samples}")

    print("\n🔮 Making predictions...")
    predictions_proba = model.predict(test_generator, verbose=1)
    y_pred = (predictions_proba > 0.5).astype(int).flatten()
    y_true = test_generator.classes

    class_names = list(test_generator.class_indices.keys())

    accuracy = np.mean(y_pred == y_true)

    print("\n" + "="*70)
    print(f"📊 Test Accuracy: {accuracy*100:.2f}%")
    print("="*70)

    # Classification report
    print("\n📋 Classification Report:")
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)

    # Save report
    report_path = f'{RESULTS_DIR}/{model_type}_classification_report.txt'
    with open(report_path, 'w') as f:
        f.write(f"{model_type.upper()} MODEL - CLASSIFICATION REPORT\n")
        f.write("="*70 + "\n\n")
        f.write(f"Test Accuracy: {accuracy*100:.2f}%\n\n")
        f.write(report)
    print(f"✅ Report saved: {report_path}")

    # Generate plots
    print("\n📊 Generating visualizations...")
    plot_training_history(model_type)
    plot_confusion_matrix(y_true, y_pred, model_type, class_names)
    roc_auc = plot_roc_curve(y_true, predictions_proba.flatten(), model_type)
    plot_sample_predictions(model, test_generator, model_type)

    print(f"\n✅ Evaluation complete for {model_type.upper()}")

    return {
        'model_type': model_type,
        'accuracy': accuracy,
        'roc_auc': roc_auc
    }


def create_comparison(results):
    """Create comparison plots"""
    print("\n" + "="*70)
    print("CREATING MODEL COMPARISON")
    print("="*70)

    models = [r['model_type'].upper() for r in results]
    accuracies = [r['accuracy'] * 100 for r in results]
    aucs = [r['roc_auc'] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    colors = ['#3498db', '#2ecc71', '#e74c3c']

    # Accuracy comparison
    bars1 = ax1.bar(models, accuracies, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 100])
    ax1.grid(axis='y', alpha=0.3)

    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%', ha='center', va='bottom', fontweight='bold')

    # AUC comparison
    bars2 = ax2.bar(models, aucs, color=colors, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('AUC Score', fontsize=12, fontweight='bold')
    ax2.set_title('Model AUC Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylim([0, 1])
    ax2.grid(axis='y', alpha=0.3)

    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    save_path = f'{RESULTS_DIR}/model_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    plt.close()

    # Summary
    summary_path = f'{RESULTS_DIR}/final_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("FACE MASK DETECTION - FINAL SUMMARY\n")
        f.write("="*70 + "\n\n")

        results_sorted = sorted(results, key=lambda x: x['accuracy'], reverse=True)

        for i, r in enumerate(results_sorted, 1):
            f.write(f"{i}. {r['model_type'].upper()}\n")
            f.write(f"   Accuracy: {r['accuracy']*100:.2f}%\n")
            f.write(f"   AUC: {r['roc_auc']:.4f}\n\n")

        best = results_sorted[0]
        f.write(f"\nBEST MODEL: {best['model_type'].upper()}\n")
        f.write(f"   Accuracy: {best['accuracy']*100:.2f}%\n")

    print(f"✅ Summary saved: {summary_path}")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    results = []

    for model_type in MODELS_TO_EVALUATE:
        result = evaluate_model(model_type)
        if result:
            results.append(result)

    if results:
        create_comparison(results)

        print("\n" + "="*70)
        print("✅ ALL EVALUATIONS COMPLETE!")
        print("="*70)
        print(f"\nFiles generated in {RESULTS_DIR}/")
        print("  Ready for Person 5 to write the report!")
        print("="*70)