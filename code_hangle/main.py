# main.py (Evaluation Metrics Added)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from model import OCRModel
from dataset import OCRDataset, is_valid_hangul_sample # Import the filter function
from loss import CTCLoss
import os
import time
from tqdm import tqdm # Import tqdm for evaluate_model progress

# --- Add necessary imports for evaluation ---
try:
    import Levenshtein
except ImportError:
    print("Warning: python-Levenshtein not installed. CER calculation will be skipped.")
    print("Please install it: pip install python-Levenshtein")
    Levenshtein = None

try:
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
except ImportError:
    print("Warning: scikit-learn or seaborn not installed. Confusion matrix will be skipped.")
    print("Please install them: pip install scikit-learn seaborn")
    confusion_matrix = None
    sns = None
# -------------------------------------------

# --- Helper functions (from visualize.py, moved here) ---
def find_korean_font():
    font_files = fm.findSystemFonts(fontpaths=None, fontext='ttf')
    for font_file in font_files:
        if 'malgun' in font_file.lower(): return font_file
    for font_file in font_files:
        if 'apple' in font_file.lower() and 'gothic' in font_file.lower(): return font_file
    for font_file in font_files:
        if 'nanum' in font_file.lower() and 'gothic' in font_file.lower(): return font_file
    # Fallback for Linux without Nanum
    for font_file in font_files:
        if 'noto' in font_file.lower() and 'sans' in font_file.lower() and 'kr' in font_file.lower(): return font_file
    return font_files[0] if font_files else None

def ctc_decode(indices, idx_to_char):
    """Simple 'Best Path' CTC decoder."""
    text = ""
    last_char_idx = -1
    for idx in indices:
        if not isinstance(idx, (int, np.integer)): # Handle potential non-integer types
             idx = int(idx)
        if idx == 0: # 0 is the blank label
            last_char_idx = -1
            continue
        # Skip duplicate characters unless separated by blank
        if idx == last_char_idx:
            continue

        char = idx_to_char.get(idx, '?') # Use get with default '?'
        if char != '?': # Append only if valid char found
             text += char
        last_char_idx = idx
    return text
# ---------------------------------------------------------

# ----------------------------------------------------
# 1. 옵티마이저 (Adam) - (No changes needed)
# ----------------------------------------------------
class Adam:
    """Adam 옵티마이저"""
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def update(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]

        self.t += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2**self.t) / (1.0 - self.beta1**self.t)

        for i in range(len(params)):
            # Skip update if gradient is nan/inf
            if np.isnan(grads[i]).any() or np.isinf(grads[i]).any():
                continue

            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grads[i]**2)
            params[i] -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + self.epsilon)

# ----------------------------------------------------
# 2. 그래디언트 클리핑 - (No changes needed)
# ----------------------------------------------------
def clip_grads(grads, max_norm):
    """(nan/inf-proof) 그래디언트 클리핑"""
    total_norm = 0
    for i in range(len(grads)):
        # Replace nan/inf before norm calculation
        grads[i] = np.nan_to_num(grads[i], nan=0.0, posinf=1e6, neginf=-1e6)
        total_norm += np.sum(grads[i]**2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        for i in range(len(grads)):
            grads[i] *= rate

# ----------------------------------------------------
# 3. 학습 및 검증 함수 (train_model) - (Save best model logic)
# ----------------------------------------------------
def train_model(model, train_dataset, valid_dataset, epochs=20, batch_size=32, lr=0.001, model_save_path="best_model.npz"):
    """모델 학습 및 검증 + 최고 성능 모델 저장"""
    optimizer = Adam(lr=lr)
    train_losses = []
    valid_losses = []
    best_valid_loss = np.inf # Track best validation loss
    start_time = time.time()

    print("Starting training and validation...")

    for epoch in range(epochs):
        epoch_start_time = time.time()
        # --- Training Phase ---
        model.is_training = True # Set training mode (if dropout/BN were used)
        epoch_train_loss = 0
        train_batch_count = 0 # Count valid batches

        num_train_batches = len(train_dataset) // batch_size
        train_iterator = tqdm(range(num_train_batches), desc=f"Epoch {epoch+1}/{epochs} [Train]")

        for batch in train_iterator:
            x_batch, t_batch = train_dataset.get_batch(batch_size)

            # Skip batch if all labels are empty (can happen with filtering)
            if not any(len(t) > 0 for t in t_batch):
                continue

            loss = model.loss(x_batch, t_batch)

            # Check for nan/inf loss immediately
            if np.isnan(loss) or np.isinf(loss):
                print(f"\nStopping training at Epoch {epoch+1}, Batch {batch} due to nan/inf loss: {loss}")
                return train_losses, valid_losses # Stop training immediately

            model.backward()
            params, grads = model.get_params_and_grads()
            clip_grads(grads, 5.0)
            optimizer.update(params, grads)

            epoch_train_loss += loss
            train_batch_count += 1
            train_iterator.set_postfix(loss=f"{loss:.4f}") # Show loss in tqdm bar

        # Calculate average train loss for the epoch
        avg_train_loss = epoch_train_loss / train_batch_count if train_batch_count > 0 else 0
        train_losses.append(avg_train_loss)

        # --- Validation Phase ---
        model.is_training = False # Set evaluation mode
        epoch_valid_loss = 0
        valid_batch_count = 0 # Count valid batches

        num_valid_batches = len(valid_dataset) // batch_size
        valid_iterator = tqdm(range(num_valid_batches), desc=f"Epoch {epoch+1}/{epochs} [Valid]")

        for batch in valid_iterator:
            x_batch, t_batch = valid_dataset.get_batch(batch_size)

            if not any(len(t) > 0 for t in t_batch):
                continue

            loss = model.loss(x_batch, t_batch) # Only forward pass

            # Accumulate loss only if valid
            if not np.isnan(loss) and not np.isinf(loss):
                epoch_valid_loss += loss
                valid_batch_count += 1
                valid_iterator.set_postfix(loss=f"{loss:.4f}")

        # Calculate average valid loss for the epoch
        avg_valid_loss = epoch_valid_loss / valid_batch_count if valid_batch_count > 0 else np.inf # Use inf if no valid batches
        valid_losses.append(avg_valid_loss)

        epoch_duration = time.time() - epoch_start_time

        # --- Epoch Summary & Save Best Model ---
        summary = f"Epoch {epoch+1}/{epochs} Summary - Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}, Duration: {epoch_duration:.2f}s"
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            model.save_params(model_save_path)
            summary += " ✨ New Best Model Saved!"
        print(summary + "\n")

    total_duration = time.time() - start_time
    print(f"Training finished in {total_duration:.2f}s")
    return train_losses, valid_losses

# ----------------------------------------------------
# 4. 시각화 함수 (plot_results) - (No significant changes needed)
# ----------------------------------------------------
def plot_results(train_losses, valid_losses, save_path="training_results.png"):
    """학습 및 검증 결과 시각화"""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    korean_font = find_korean_font()
    font_prop = fm.FontProperties(fname=korean_font) if korean_font else None

    # Filter out potential nan/inf/0 values before plotting
    def filter_values(losses):
        return [l for i, l in enumerate(losses) if not np.isnan(l) and not np.isinf(l) and l != 0]

    epochs = range(1, len(train_losses) + 1)
    clean_train_losses = filter_values(train_losses)
    clean_valid_losses = filter_values(valid_losses)
    valid_epochs = range(1, len(clean_valid_losses) + 1) # Adjust x-axis for valid loss if needed

    if not clean_train_losses:
        print("Warning: No valid train losses to plot.")
        return

    ax1.plot(epochs[:len(clean_train_losses)], clean_train_losses, 'b-o', label='Training Loss')
    if clean_valid_losses:
        best_epoch = np.argmin(clean_valid_losses) + 1
        min_valid_loss = np.min(clean_valid_losses)
        ax1.plot(valid_epochs, clean_valid_losses, 'r-s', label=f'Validation Loss (Best: {min_valid_loss:.4f} at Epoch {best_epoch})')
        # Mark the best epoch
        ax1.plot(best_epoch, min_valid_loss, 'r*', markersize=15, label=f'Best Epoch ({best_epoch})')

    ax1.set_xlabel('Epoch', fontproperties=font_prop)
    ax1.set_ylabel('Loss', fontproperties=font_prop)
    ax1.set_title('Training & Validation Loss over Epochs', fontproperties=font_prop)
    ax1.legend(prop=font_prop)
    ax1.grid(True)
    ax1.xaxis.get_major_locator().set_params(integer=True) # Ensure integer ticks for epochs

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Loss curve saved to {os.path.abspath(save_path)}")
    plt.close(fig)

# ----------------------------------------------------
# 5. [NEW] 평가 함수 (evaluate_model)
# ----------------------------------------------------
def evaluate_model(model, dataset, batch_size=32, max_batches=None):
    """최고 성능 모델 로드 후 평가 (CER, Accuracy 계산)"""
    if Levenshtein is None:
        print("Levenshtein distance calculation skipped.")
        return None, None, None, None # Return None for metrics

    model.is_training = False # Evaluation mode
    total_edit_distance = 0
    total_true_length = 0
    exact_matches = 0
    total_samples = 0

    all_true_chars = []
    all_pred_chars = []

    num_batches = len(dataset) // batch_size
    if max_batches:
        num_batches = min(num_batches, max_batches) # Limit batches for faster evaluation if needed

    eval_iterator = tqdm(range(num_batches), desc="Evaluating Model")

    for batch in eval_iterator:
        x_batch, t_batch = dataset.get_batch(batch_size)

        # Skip batch if all labels are empty
        if not any(len(t) > 0 for t in t_batch):
            continue

        pred_indices = model.predict(x_batch) # Get model predictions (N, T)

        current_batch_size = x_batch.shape[0]
        total_samples += current_batch_size

        batch_edit_distance = 0
        batch_true_length = 0
        batch_exact_matches = 0

        for i in range(current_batch_size):
            # Decode true label sequence
            true_text = ctc_decode(t_batch[i], dataset.idx_to_char)
            # Decode predicted index sequence
            pred_text = ctc_decode(pred_indices[i], dataset.idx_to_char)

            if len(true_text) == 0 and len(pred_text) == 0:
                 # If both are empty, count as exact match, distance 0
                 batch_exact_matches += 1
                 continue
            elif len(true_text) == 0:
                 # If true is empty but pred is not, distance is len(pred)
                 edit_distance = len(pred_text)
            else:
                 # Calculate Levenshtein distance
                 edit_distance = Levenshtein.distance(pred_text, true_text)

            batch_edit_distance += edit_distance
            batch_true_length += len(true_text)

            if pred_text == true_text:
                batch_exact_matches += 1

            # Store characters for confusion matrix (use true length for alignment)
            # This alignment is approximate for CTC, good for overall stats
            min_len = min(len(pred_text), len(true_text))
            all_true_chars.extend(list(true_text[:min_len]))
            all_pred_chars.extend(list(pred_text[:min_len]))


        total_edit_distance += batch_edit_distance
        total_true_length += batch_true_length
        exact_matches += batch_exact_matches

        # Update tqdm description with running CER/Acc
        current_cer = (total_edit_distance / total_true_length) * 100 if total_true_length > 0 else 0
        current_acc = (exact_matches / total_samples) * 100 if total_samples > 0 else 0
        eval_iterator.set_postfix(CER=f"{current_cer:.2f}%", Acc=f"{current_acc:.2f}%")

    # Final calculations
    final_cer = (total_edit_distance / total_true_length) * 100 if total_true_length > 0 else 0
    final_accuracy = (exact_matches / total_samples) * 100 if total_samples > 0 else 0

    print(f"\nEvaluation Complete:")
    print(f"  - Total Samples Evaluated: {total_samples}")
    print(f"  - Character Error Rate (CER): {final_cer:.2f}%")
    print(f"  - Exact Match Accuracy: {final_accuracy:.2f}%")

    return final_cer, final_accuracy, all_true_chars, all_pred_chars

# ----------------------------------------------------
# 6. [NEW] Confusion Matrix 시각화 함수
# ----------------------------------------------------
def plot_confusion_matrix(true_chars, pred_chars, dataset, top_n=30, save_path="confusion_matrix.png"):
    """선택된 문자들에 대한 Confusion Matrix 시각화"""
    if confusion_matrix is None or sns is None:
        print("Confusion matrix plotting skipped.")
        return

    korean_font = find_korean_font()
    font_prop = fm.FontProperties(fname=korean_font) if korean_font else None
    plt.rc('font', family=font_prop.get_name() if font_prop else 'sans-serif') # Set default font


    # Get all unique characters present in the predictions/truths
    unique_chars = sorted(list(set(true_chars) | set(pred_chars)))

    if not unique_chars:
        print("Warning: No characters found to plot confusion matrix.")
        return

    # Select top N most frequent characters from the true labels for display
    from collections import Counter
    char_counts = Counter(true_chars)
    top_chars = [char for char, count in char_counts.most_common(top_n)]
    labels = sorted(list(set(top_chars))) # Use only top N for matrix labels

    if not labels:
        print(f"Warning: Could not determine top {top_n} characters.")
        return

    print(f"Generating Confusion Matrix for Top {len(labels)} frequent characters...")

    # Filter predictions and truths to only include top N chars
    filtered_true = [c for c in true_chars if c in labels]
    filtered_pred = [pred_chars[i] for i, true_char in enumerate(true_chars) if true_char in labels]
    # Ensure pred_chars corresponding to filtered true_chars are kept, even if pred is not in top N
    filtered_pred_aligned = []
    original_pred_idx = 0
    for i in range(len(true_chars)):
        if true_chars[i] in labels:
            # Need to handle index carefully if pred_chars was shorter
            if original_pred_idx < len(pred_chars):
                 filtered_pred_aligned.append(pred_chars[original_pred_idx])
            else: # Should not happen with min_len logic, but as safeguard
                 filtered_pred_aligned.append('?') # Or some placeholder
        if original_pred_idx < len(pred_chars):
            original_pred_idx += 1


    # Calculate confusion matrix using sklearn
    # Ensure labels includes all chars present in filtered_pred_aligned too, up to a limit
    all_present_labels = sorted(list(set(filtered_true) | set(filtered_pred_aligned)))
    # Limit labels to top_n + potentially a few others that appeared in predictions
    cm_labels = sorted(list(set(labels) | set(all_present_labels)))


    # Filter again based on the final cm_labels list
    final_true = []
    final_pred = []
    for t, p in zip(true_chars, pred_chars): # Iterate original full lists
        if t in cm_labels: # If true char is one we want to show
            final_true.append(t)
            final_pred.append(p if p in cm_labels else '?') # If pred is outside, mark as '?'

    if not final_true:
         print("No data left after filtering for confusion matrix labels.")
         return

    # Add '?' to labels if it occurred
    if '?' in final_pred and '?' not in cm_labels:
         cm_labels.append('?')
         cm_labels.sort()


    cm = confusion_matrix(final_true, final_pred, labels=cm_labels)

    # Plotting the confusion matrix using seaborn
    plt.figure(figsize=(max(10, len(cm_labels)//2), max(8, len(cm_labels)//2))) # Adjust size based on labels
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=cm_labels, yticklabels=cm_labels, annot_kws={"size": 8})
    plt.xlabel('Predicted Label', fontproperties=font_prop)
    plt.ylabel('True Label', fontproperties=font_prop)
    plt.title(f'Confusion Matrix (Top {len(labels)} Frequent Characters)', fontproperties=font_prop)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Confusion matrix saved to {os.path.abspath(save_path)}")
    plt.close()


# ----------------------------------------------------
# 7. 메인 함수 (main)
# ----------------------------------------------------
def main():
    """메인 실행 함수"""
    # 하이퍼파라미터
    EPOCHS = 20
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-5 # 0.00001
    MAX_SAMPLES_TRAIN = 10000
    MAX_SAMPLES_VALID = 2000 # Use more samples for better evaluation

    # --- Select Dataset Type ---
    USE_HANGUL_FILTER = True # ★★★ True로 바꾸면 한글 전용으로 실행 ★★★
    # ---------------------------

    if USE_HANGUL_FILTER:
        print("Using Hangul-only dataset filtering.")
        MODEL_SAVE_FILENAME = "best_hangul_ocr_model.npz"
        LOSS_PLOT_FILENAME = "training_hangul_results.png"
        CONFUSION_MATRIX_FILENAME = "confusion_matrix_hangul.png"
        PREDICTION_PLOT_FILENAME = "prediction_hangul_results.png"
        VOCAB_FILENAME = "hangul_vocab.json" # Expected vocab file name
    else:
        print("Using full dataset (including Hangul, Hanja, Eng, etc.).")
        MODEL_SAVE_FILENAME = "best_ocr_model.npz"
        LOSS_PLOT_FILENAME = "training_results.png"
        CONFUSION_MATRIX_FILENAME = "confusion_matrix.png"
        PREDICTION_PLOT_FILENAME = "prediction_results.png"
        VOCAB_FILENAME = "vocab.json" # Expected vocab file name

    TRAIN_DATA_DIR = "data/train"
    VALID_DATA_DIR = "data/valid"
    MODEL_SAVE_PATH = os.path.join("data", MODEL_SAVE_FILENAME) # Save in 'data' folder
    VOCAB_PATH = os.path.join("data", VOCAB_FILENAME) # Vocab path

    print("=" * 60)
    print(" OCR Deep Learning Project - NumPy Implementation (CRNN-CTC)")
    print(f" Mode: {'Hangul Only' if USE_HANGUL_FILTER else 'Full Dataset'}")
    print("=" * 60)

    # --- Dataset Loading ---
    print("\n[1] Loading datasets...")
    if not os.path.exists(TRAIN_DATA_DIR):
        print(f"Fatal: Training directory '{TRAIN_DATA_DIR}' not found. Run preprocess.py first!")
        return
    # Pass USE_HANGUL_FILTER to dataset loader if it supports it
    # Assuming the provided dataset.py has the filtering integrated based on its internal logic
    train_dataset = OCRDataset(TRAIN_DATA_DIR, max_samples=MAX_SAMPLES_TRAIN, hangul_only=USE_HANGUL_FILTER)
    num_classes = train_dataset.get_vocab_size()

    if not os.path.exists(VALID_DATA_DIR):
        print(f"Fatal: Validation directory '{VALID_DATA_DIR}' not found.")
        return
    # Load validation dataset with the same filter setting
    valid_dataset = OCRDataset(VALID_DATA_DIR, max_samples=MAX_SAMPLES_VALID, hangul_only=USE_HANGUL_FILTER)
    # Crucially, ensure validation set uses the vocabulary built from the training set
    # Load the correct vocab file
    if os.path.exists(VOCAB_PATH):
         print(f"Loading vocabulary for validation set from {VOCAB_PATH}...")
         try:
             with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
                 data = json.load(f)
                 valid_dataset.char_to_idx = data['char_to_idx']
                 valid_dataset.idx_to_char = {int(k): v for k, v in data['idx_to_char'].items()}
             print("Vocabulary loaded successfully for validation set.")
             # Re-check num_classes consistency
             if num_classes != len(valid_dataset.char_to_idx):
                  print(f"Warning: Discrepancy in vocabulary size! Train: {num_classes}, Loaded for Valid: {len(valid_dataset.char_to_idx)}")
                  num_classes = len(valid_dataset.char_to_idx) # Use loaded vocab size
         except Exception as e:
             print(f"Error loading vocabulary from {VOCAB_PATH}: {e}. Validation might fail.")
             return
    else:
         print(f"Error: Vocabulary file '{VOCAB_PATH}' not found. Cannot proceed.")
         return


    print(f"\n[2] Creating model... (NumClasses = {num_classes})")
    model = OCRModel(num_classes=num_classes)
    model.loss_layer = CTCLoss(blank_label=0) # Make sure blank label matches dataset

    print("\n[3] Training model...")
    train_losses, valid_losses = train_model(
        model, train_dataset, valid_dataset,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        model_save_path=MODEL_SAVE_PATH
    )

    print("\n[4] Plotting training results...")
    plot_results(train_losses, valid_losses, save_path=LOSS_PLOT_FILENAME)

    print("\n[5] Evaluating best model...")
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"Loading best model from {MODEL_SAVE_PATH}...")
        try:
            model.load_params(MODEL_SAVE_PATH)
            # Use the actual validation dataset loaded earlier
            cer, accuracy, true_chars, pred_chars = evaluate_model(model, valid_dataset, BATCH_SIZE)

            if cer is not None and true_chars is not None: # Check if evaluation ran
                print("\n[6] Plotting confusion matrix...")
                plot_confusion_matrix(true_chars, pred_chars, valid_dataset, top_n=30, save_path=CONFUSION_MATRIX_FILENAME)

                print("\n[7] Visualizing predictions from best model...")
                # Call prediction visualization using the loaded best model
                show_predictions_main(model, valid_dataset, num_samples=5, save_path=PREDICTION_PLOT_FILENAME)

        except Exception as e:
            print(f"Error during evaluation or visualization: {e}")
            import traceback
            traceback.print_exc()

    else:
        print(f"Warning: Best model file '{MODEL_SAVE_PATH}' not found. Evaluation and prediction visualization skipped.")
        print("Running prediction visualization with the *last* epoch's (potentially diverged) model state as an example.")
        # Optionally run visualization with the model state at the end of training
        show_predictions_main(model, valid_dataset, num_samples=5, save_path=PREDICTION_PLOT_FILENAME.replace('.png', '_last_epoch.png'))


    print("\n" + "=" * 60)
    print(" Process Completed!")
    print(f" Best model saved to: {os.path.abspath(MODEL_SAVE_PATH) if os.path.exists(MODEL_SAVE_PATH) else 'Not saved'}")
    print(f" Loss curve saved to: {os.path.abspath(LOSS_PLOT_FILENAME)}")
    if cer is not None:
         print(f" Confusion matrix saved to: {os.path.abspath(CONFUSION_MATRIX_FILENAME)}")
         print(f" Prediction examples saved to: {os.path.abspath(PREDICTION_PLOT_FILENAME)}")
         print(f" Final CER: {cer:.2f}%, Final Accuracy: {accuracy:.2f}%")
    print("=" * 60)

# ----------------------------------------------------
# 8. [NEW] Prediction Visualization Function (from visualize.py)
# ----------------------------------------------------
# This function is now part of main.py to be called after loading the best model
def show_predictions_main(model, dataset, num_samples=5, save_path="prediction_results.png"):
    """최고 성능 모델로 예측 결과 시각화"""
    print(f"\n Generating {num_samples} prediction examples...")

    korean_font = find_korean_font()
    font_prop = fm.FontProperties(fname=korean_font, size=10) if korean_font else None
    plt.rc('font', family=font_prop.get_name() if font_prop else 'sans-serif')

    # Get random samples ensuring unique indices if dataset size allows
    num_available = len(dataset)
    if num_samples > num_available:
        print(f"Warning: Requested {num_samples} samples, but only {num_available} available in dataset.")
        num_samples = num_available
        indices = np.arange(num_available)
    else:
        indices = np.random.choice(num_available, num_samples, replace=False)

    images = []
    true_texts = []
    pred_texts = []

    print("Fetching samples and predicting...")
    samples_fetched = 0
    attempts = 0
    max_attempts = num_available * 2 # Safety break
    processed_indices = set()

    # Ensure we get exactly num_samples with valid labels
    while samples_fetched < num_samples and attempts < max_attempts:
        idx = np.random.choice(num_available)
        if idx in processed_indices:
             attempts += 1
             continue
        processed_indices.add(idx)
        attempts += 1

        img, label_indices = dataset[idx] # Get single sample
        true_text = ctc_decode(label_indices, dataset.idx_to_char)

        # Skip if true text is empty, as it's less informative
        if not true_text:
            continue

        # Predict using the model (requires batch dimension)
        img_batch = img.reshape(1, 1, 32, 32) # Add batch and channel dims
        pred_indices_single = model.predict(img_batch)[0] # Predict and get first item
        pred_text = ctc_decode(pred_indices_single, dataset.idx_to_char)

        images.append(img.reshape(32, 32))
        true_texts.append(true_text)
        pred_texts.append(pred_text)
        samples_fetched += 1


    if samples_fetched == 0:
         print("Could not fetch any valid samples for prediction visualization.")
         return

    # Plotting
    num_plot = len(images) # Number of successfully fetched samples
    fig, axes = plt.subplots(1, num_plot, figsize=(num_plot * 2.5, 3.5))
    if num_plot == 1: axes = [axes]

    print("Plotting predictions...")
    for i in range(num_plot):
        axes[i].imshow(images[i], cmap='gray')
        axes[i].axis('off')
        title = f"True: {true_texts[i]}\nPred: {pred_texts[i]}"
        # Calculate CER for this sample if Levenshtein is available
        if Levenshtein and len(true_texts[i]) > 0:
            dist = Levenshtein.distance(pred_texts[i], true_texts[i])
            sample_cer = (dist / len(true_texts[i])) * 100
            title += f"\n(CER: {sample_cer:.1f}%)"
        elif Levenshtein and len(true_texts[i]) == 0 and len(pred_texts[i]) > 0:
             sample_cer = float('inf') # Or handle as 100% error
             title += f"\n(CER: inf)"
        elif Levenshtein and len(true_texts[i]) == 0 and len(pred_texts[i]) == 0:
             sample_cer = 0.0
             title += f"\n(CER: 0.0%)"


        axes[i].set_title(title, fontproperties=font_prop, fontsize=9)

    plt.tight_layout(pad=0.5)
    plt.savefig(save_path, dpi=300)
    print(f"Prediction examples saved to {os.path.abspath(save_path)}")
    plt.close(fig)


# ----------------------------------------------------
# 9. 메인 실행 블록
# ----------------------------------------------------
if __name__ == "__main__":
    # Add a check for dataset.py supporting hangul_only flag
    try:
        # Check if OCRDataset accepts 'hangul_only' argument
        # This is a bit hacky, relies on inspecting the init signature
        import inspect
        sig = inspect.signature(OCRDataset.__init__)
        if 'hangul_only' not in sig.parameters:
            print("="*60)
            print("ERROR: Your dataset.py does not seem to support the 'hangul_only' flag.")
            print("Please update dataset.py to the version with Hangul filtering.")
            print("="*60)
            exit() # Stop execution if dataset.py is not updated
        import json # Need json for vocab loading check in main
    except ImportError:
         pass # Let the main function handle dataset loading errors

    main()

