"""
Skin Lesion Classification — Colab-Safe Version (Repair-Split Fix)
Handles partially-created dataset_split by rebuilding val from train.
"""


# ── CELL 1: Imports ────────────────────────────────────────────────────────────
import os, glob, shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

print("TF version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))

# ── CELL 2: CONFIG ─────────────────────────────────────────────────────────────
IMG_SIZE    = 128
INPUT_SHAPE = (128, 128, 3)
BATCH_SIZE  = 16
EPOCHS      = 50
LR_MAX      = 0.001
LR_MIN      = 1e-6
TTA_STEPS   = 5
AUTOTUNE    = tf.data.AUTOTUNE

DATASET_DIR = "D:/mini_project_sahyadri/train"   # original images
SPLIT_DIR   = "D:/mini_project_sahyadri/dataset_split"   # already has train/ and test/

SKIN_CLASSES = [
    'Acne', 'Actinic Keratosis', 'Basal Cell Carcinoma',
    'Chickenpox', 'Dermato Fibroma', 'Dyshidrotic Eczema',
    'Melanoma', 'Nail Fungus', 'Nevus', 'Normal Skin',
    'Pigmented Benign Keratosis', 'Ringworm', 'Seborrheic Keratosis',
    'Squamous Cell Carcinoma', 'Vascular Lesion',
]
NUM_CLASSES = len(SKIN_CLASSES)
CLASS_INDEX = {cls: i for i, cls in enumerate(SKIN_CLASSES)}
print(f"Classes: {NUM_CLASSES}")


# ── CELL 3: REPAIR — carve val from existing train split ──────────────────────
def repair_val_split(split_dir, val_ratio=0.15, seed=42):
    """
    If val/ is missing or empty, moves ~15% of train/ images into val/.
    Safe to re-run — skips classes that already have val images.
    """
    print("── Checking / repairing val split ──")
    for cls in SKIN_CLASSES:
        train_cls = os.path.join(split_dir, 'train', cls)
        val_cls   = os.path.join(split_dir, 'val',   cls)
        os.makedirs(val_cls, exist_ok=True)

        existing_val = glob.glob(os.path.join(val_cls, "*.*"))
        if len(existing_val) > 0:
            print(f"  {cls}: val already has {len(existing_val)} files — skipping")
            continue

        train_imgs = (glob.glob(os.path.join(train_cls, "*.jpg"))  +
                      glob.glob(os.path.join(train_cls, "*.jpeg")) +
                      glob.glob(os.path.join(train_cls, "*.png")))

        if not train_imgs:
            print(f"  [WARN] {cls}: no train images found at {train_cls}")
            continue

        keep, move_to_val = train_test_split(
            train_imgs, test_size=val_ratio, random_state=seed
        )

        for p in move_to_val:
            shutil.move(p, val_cls)   # move (not copy) to keep disk usage same

        print(f"  {cls}: moved {len(move_to_val)} → val  |  {len(keep)} remain in train")

    print("Val split repair done.\n")

repair_val_split(SPLIT_DIR)


# ── CELL 4: VERIFY counts ──────────────────────────────────────────────────────
print("── Split counts ──")
total = {'train': 0, 'val': 0, 'test': 0}
for split in ('train', 'val', 'test'):
    for cls in SKIN_CLASSES:
        cls_dir = os.path.join(SPLIT_DIR, split, cls)
        n = len(glob.glob(os.path.join(cls_dir, "*.*")))
        total[split] += n
for split, n in total.items():
    print(f"  {split}: {n} total files")
    if n == 0:
        raise RuntimeError(
            f"'{split}' split still empty after repair!\n"
            f"Check that {os.path.join(SPLIT_DIR, split)} exists and has sub-folders."
        )
print()


# ── CELL 5: tf.data PIPELINE ───────────────────────────────────────────────────
def get_file_paths_and_labels(split):
    paths, labels = [], []
    for cls in SKIN_CLASSES:
        cls_dir = os.path.join(SPLIT_DIR, split, cls)
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG'):
            for p in glob.glob(os.path.join(cls_dir, ext)):
                paths.append(p)
                labels.append(CLASS_INDEX[cls])
    return paths, labels

def parse_image(path, label, augment=False):
    raw = tf.io.read_file(path)
    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0
    if augment:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_brightness(img, 0.15)
        img = tf.image.random_contrast(img, 0.85, 1.15)
        img = tf.clip_by_value(img, 0.0, 1.0)
    return img, tf.one_hot(label, NUM_CLASSES)

def make_dataset(split, augment=False, shuffle=False):
    paths, labels = get_file_paths_and_labels(split)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=42)
    ds = ds.map(lambda p, l: parse_image(p, l, augment=augment),
                num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds, len(paths)

print("Building datasets …")
train_ds, n_train = make_dataset('train', augment=True,  shuffle=True)
val_ds,   n_val   = make_dataset('val',   augment=False, shuffle=False)
test_ds,  n_test  = make_dataset('test',  augment=False, shuffle=False)
print(f"✓ Train: {n_train}  |  Val: {n_val}  |  Test: {n_test}")


# ── CELL 6: MODEL ──────────────────────────────────────────────────────────────
def create_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
    return models.Sequential([
        layers.Conv2D(32, 3, activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(), layers.Dropout(0.25),

        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(), layers.Dropout(0.25),

        layers.Conv2D(128, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(), layers.Dropout(0.25),

        layers.Conv2D(256, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(), layers.Dropout(0.25),

        layers.Conv2D(512, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(), layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(1024, activation='relu'),
        layers.BatchNormalization(), layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(), layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax'),
    ])

model = create_model()
model.compile(optimizer=Adam(LR_MAX),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()


# ── CELL 7: LR SCHEDULE ────────────────────────────────────────────────────────
def cosine_annealing(epoch, _=None):
    return float(LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + np.cos(np.pi * epoch / EPOCHS)))


# ── CELL 8: TRAIN ──────────────────────────────────────────────────────────────
callbacks = [
    EarlyStopping(monitor='val_loss', patience=15,
                  restore_best_weights=True, verbose=1),
    ModelCheckpoint('best_skin_lesion_model.h5',
                    monitor='val_accuracy', save_best_only=True, verbose=1),
    LearningRateScheduler(cosine_annealing, verbose=0),
]

print("\n── Training ──")
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=callbacks,
)


# ── CELL 9: EVALUATE ───────────────────────────────────────────────────────────
print("\n── Evaluation ──")
test_loss, test_acc = model.evaluate(test_ds, verbose=0)
print(f"Test Accuracy : {test_acc:.4f}")
print(f"Test Loss     : {test_loss:.4f}")


# ── CELL 10: TTA ───────────────────────────────────────────────────────────────
print(f"\n── TTA ({TTA_STEPS} passes) ──")

def predict_with_tta(model, test_paths, test_labels_int, tta_steps=TTA_STEPS):
    preds = np.zeros((len(test_paths), NUM_CLASSES), dtype=np.float32)
    for i in range(tta_steps):
        print(f"  pass {i+1}/{tta_steps} …")
        ds = tf.data.Dataset.from_tensor_slices((test_paths, test_labels_int))
        ds = ds.map(lambda p, l: parse_image(p, l, augment=True),
                    num_parallel_calls=AUTOTUNE)
        ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
        preds += model.predict(ds, verbose=0)
    return preds / tta_steps

test_paths, test_labels_int = get_file_paths_and_labels('test')
y_pred_probs = predict_with_tta(model, test_paths, test_labels_int)
y_pred_tta   = np.argmax(y_pred_probs, axis=1)
y_true       = np.array(test_labels_int)

print(f"TTA Accuracy  : {np.mean(y_pred_tta == y_true):.4f}")
print(classification_report(y_true, y_pred_tta, target_names=SKIN_CLASSES))


# ── CELL 11: GRAPHS ────────────────────────────────────────────────────────────

# Graph 1 — Accuracy
plt.figure(figsize=(9, 5))
plt.plot(history.history['accuracy'],     label='Train Accuracy',      linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2, linestyle='--')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epoch'); plt.ylabel('Accuracy')
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig('graph_accuracy.png', dpi=150); plt.show()

# Graph 2 — Loss
plt.figure(figsize=(9, 5))
plt.plot(history.history['loss'],     label='Train Loss',      linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2, linestyle='--')
plt.title('Training vs Validation Loss')
plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig('graph_loss.png', dpi=150); plt.show()

# Graph 3 — Confusion Matrix
cm = confusion_matrix(y_true, y_pred_tta)
plt.figure(figsize=(16, 13))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=SKIN_CLASSES)
disp.plot(cmap='Blues', ax=plt.gca(), colorbar=False)
plt.title('Confusion Matrix (with TTA)')
plt.xticks(rotation=45, ha='right'); plt.tight_layout()
plt.savefig('graph_confusion_matrix.png', dpi=150); plt.show()

# Graph 4 — Class Distribution
class_counts = []
for cls in SKIN_CLASSES:
    cls_dir = os.path.join(DATASET_DIR, cls)
    n = (len(glob.glob(os.path.join(cls_dir, "*.jpg")))  +
         len(glob.glob(os.path.join(cls_dir, "*.jpeg"))) +
         len(glob.glob(os.path.join(cls_dir, "*.png"))))
    class_counts.append(n)
plt.figure(figsize=(14, 6))
bars = plt.bar(SKIN_CLASSES, class_counts, color='steelblue', edgecolor='black')
plt.title('Class Distribution in Dataset')
plt.xlabel('Class'); plt.ylabel('Number of Images')
plt.xticks(rotation=45, ha='right')
for bar, count in zip(bars, class_counts):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()+2,
             str(count), ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig('graph_class_distribution.png', dpi=150); plt.show()

# Graph 5 — Prediction Distribution
unique, counts = np.unique(y_pred_tta, return_counts=True)
plt.figure(figsize=(10, 10))
plt.pie(counts, labels=[SKIN_CLASSES[i] for i in unique],
        autopct='%1.1f%%', startangle=140)
plt.title('Prediction Distribution on Test Set (TTA)')
plt.tight_layout()
plt.savefig('graph_prediction_distribution.png', dpi=150); plt.show()

# Graph 6 — Per-Class Accuracy
per_class_acc = cm.diagonal() / cm.sum(axis=1)
plt.figure(figsize=(14, 6))
bars = plt.bar(SKIN_CLASSES, per_class_acc * 100,
               color='mediumseagreen', edgecolor='black')
plt.title('Per-Class Accuracy (with TTA)')
plt.xlabel('Class'); plt.ylabel('Accuracy (%)')
plt.ylim(0, 115); plt.xticks(rotation=45, ha='right')
for bar, val in zip(bars, per_class_acc):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()+1,
             f'{val*100:.1f}%', ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig('graph_per_class_accuracy.png', dpi=150); plt.show()

# Graph 7 — LR Schedule
lr_values = [cosine_annealing(e) for e in range(EPOCHS)]
plt.figure(figsize=(9, 4))
plt.plot(lr_values, color='darkorange', linewidth=2)
plt.title('Cosine Annealing Learning Rate Schedule')
plt.xlabel('Epoch'); plt.ylabel('Learning Rate')
plt.grid(True); plt.tight_layout()
plt.savefig('graph_lr_schedule.png', dpi=150); plt.show()


# ── CELL 12: SAVE ──────────────────────────────────────────────────────────────
model.save("skin_lesion_classification_model.h5")
print("\nDone! Model saved → skin_lesion_classification_model.h5")
print("Graphs: graph_accuracy.png | graph_loss.png | graph_confusion_matrix.png")
print("        graph_class_distribution.png | graph_prediction_distribution.png")
print("        graph_per_class_accuracy.png | graph_lr_schedule.png")
