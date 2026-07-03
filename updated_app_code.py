# ============================================================
# app.py
# Skin Disease Prediction with Explainable AI (Grad-CAM)
# + Groq AI Dynamic Explanation
# ============================================================

import os
import cv2
import numpy as np
import tensorflow as tf

from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

# ============================================================
# Flask Configuration
# ============================================================

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
HEATMAP_FOLDER = "static/heatmaps"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["HEATMAP_FOLDER"] = HEATMAP_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(HEATMAP_FOLDER, exist_ok=True)

# ============================================================
# Groq API Key
# ============================================================

GROQ_API_KEY = "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"

# ============================================================
# Load Model
# ============================================================

MODEL_PATH = r"D:/mini_project_sahyadri/outputs/skin_lesion_classification_model.h5"

model = load_model(MODEL_PATH)

print("Model Loaded Successfully")
print("Model Type:", model.__class__.__name__)

# ============================================================
# MUST call model on dummy input BEFORE anything else.
# ============================================================

dummy = tf.zeros((1, 128, 128, 3), dtype=tf.float32)
_ = model(dummy, training=False)

print("Model built. Input shape :", model.inputs[0].shape)
print("Model built. Output shape:", model.outputs[0].shape)

# ============================================================
# AUTO-DETECT Last Conv Layer (fixes Conv grads None issue)
# ============================================================

def get_last_conv_layer(model):
    """Automatically finds the last Conv2D layer in the model."""
    last_conv = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv = layer.name
    return last_conv

LAST_CONV_LAYER_NAME = get_last_conv_layer(model)

print("\n========== MODEL LAYERS ==========\n")
for i, layer in enumerate(model.layers):
    try:
        out_shape = layer.output.shape
    except:
        out_shape = "N/A"
    marker = " <-- LAST CONV ✅" if layer.name == LAST_CONV_LAYER_NAME else ""
    print(f"  [{i:>3}]  {layer.__class__.__name__:25s}  {layer.name:35s}  {out_shape}{marker}")

print(f"\n✅ Auto-detected Last Conv Layer: {LAST_CONV_LAYER_NAME}\n")

# ============================================================
# Build Grad-CAM Sub-model (FIXED for Sequential models)
# ============================================================

def build_grad_model(model, last_conv_layer_name):
    """
    Builds Grad-CAM sub-model correctly for Sequential models.
    Uses model.inputs[0] instead of model.input to avoid
    AttributeError on Sequential models that haven't been called.
    """
    # Build model explicitly with input shape first
    model.build((None, 128, 128, 3))

    # Use model.inputs[0] — works for both Sequential & Functional
    grad_model = tf.keras.models.Model(
        inputs=model.inputs[0],
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.outputs[0]
        ]
    )
    return grad_model

grad_model = build_grad_model(model, LAST_CONV_LAYER_NAME)
print("✅ Grad-CAM sub-model built successfully\n")

# ============================================================
# Class Names
# ============================================================

class_names = [
    "Acne",
    "Actinic Keratosis",
    "Basal Cell Carcinoma",
    "Chickenpox",
    "Dermato Fibroma",
    "Dyshidrotic Eczema",
    "Melanoma",
    "Nail Fungus",
    "Nevus",
    "Normal Skin",
    "Pigmented Benign Keratosis",
    "Ringworm",
    "Seborrheic Keratosis",
    "Squamous Cell Carcinoma",
    "Vascular Lesion"
]

# ============================================================
# Prediction Function
# ============================================================

def predict_image(img_path):
    try:
        img       = image.load_img(img_path, target_size=(128, 128))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

        prediction = model.predict(img_array, verbose=0)

        print("\n========== PREDICTIONS ==========\n")
        for i, prob in enumerate(prediction[0]):
            print(f"  {class_names[i]:35s}: {prob * 100:.2f}%")

        predicted_index = int(np.argmax(prediction))
        predicted_class = class_names[predicted_index]

        print(f"\n✅ Predicted: {predicted_class} (index: {predicted_index})")
        return predicted_class, predicted_index, img_array

    except Exception as e:
        import traceback
        print("Prediction Error:", e)
        traceback.print_exc()
        return "Error Processing Image", None, None


# ============================================================
# Grad-CAM Function (FIXED - No more Conv grads None)
# ============================================================

def generate_gradcam(img_path, img_array, pred_index):
    try:
        # -------------------------------------------------------
        # KEY FIX: Use tf.Variable instead of tf.constant
        # tf.Variable is automatically watched by GradientTape
        # tf.constant requires manual tape.watch()
        # -------------------------------------------------------
        img_tensor = tf.Variable(img_array, dtype=tf.float32)

        with tf.GradientTape() as tape:
            # tape automatically watches tf.Variable
            conv_outputs, predictions = grad_model(img_tensor, training=False)
            loss = predictions[:, pred_index]

        # Compute gradients of loss w.r.t conv layer output
        grads = tape.gradient(loss, conv_outputs)

        if grads is None:
            print("⚠️  Standard Grad-CAM failed — trying alternative method...")
            heatmap = fallback_saliency(img_array, pred_index)
        else:
            print("✅ Standard Grad-CAM gradients computed successfully")
            heatmap = compute_gradcam_heatmap(conv_outputs, grads)

        if heatmap is None:
            print("❌ All heatmap methods failed.")
            return None

        # Overlay heatmap on original image
        save_path = overlay_heatmap(img_path, heatmap)
        print("✅ GradCAM saved to:", save_path)
        return save_path

    except Exception as e:
        import traceback
        print("GradCAM Error:", e)
        traceback.print_exc()
        return None


def compute_gradcam_heatmap(conv_outputs, grads):
    """Standard Grad-CAM heatmap computation."""
    try:
        # Average gradients across spatial dimensions
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weight conv outputs by pooled gradients
        conv_outputs_0 = conv_outputs[0]
        heatmap = conv_outputs_0 @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # Apply ReLU and normalize
        heatmap = tf.maximum(heatmap, 0)
        heatmap = heatmap / (tf.math.reduce_max(heatmap) + 1e-8)
        heatmap = heatmap.numpy()
        heatmap = cv2.resize(heatmap, (128, 128))

        print("✅ Standard Grad-CAM heatmap computed")
        return heatmap

    except Exception as e:
        print("Heatmap computation error:", e)
        return None


def fallback_saliency(img_array, pred_index):
    """Fallback: Input gradient saliency map."""
    try:
        img_tensor = tf.Variable(img_array, dtype=tf.float32)

        with tf.GradientTape() as tape:
            preds = model(img_tensor, training=False)
            loss  = preds[:, pred_index]

        input_grads = tape.gradient(loss, img_tensor)

        if input_grads is None:
            print("❌ Saliency map also failed.")
            return None

        saliency = tf.reduce_max(tf.abs(input_grads), axis=-1)[0].numpy()
        heatmap  = cv2.resize(saliency, (128, 128))
        heatmap  = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        print("✅ Fallback saliency map generated")
        return heatmap

    except Exception as e:
        print("Saliency error:", e)
        return None


def overlay_heatmap(img_path, heatmap):
    """Overlay colored heatmap on original image and save."""
    try:
        img_bgr         = cv2.imread(img_path)
        img_bgr         = cv2.resize(img_bgr, (128, 128))
        heatmap_uint8   = np.uint8(255 * heatmap)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        superimposed    = cv2.addWeighted(img_bgr, 0.6, heatmap_colored, 0.4, 0)

        save_path = os.path.join(app.config["HEATMAP_FOLDER"], "gradcam.jpg")
        cv2.imwrite(save_path, superimposed)
        return save_path

    except Exception as e:
        print("Overlay error:", e)
        return None


# ============================================================
# Groq AI Explanation Function
# ============================================================

def get_ai_explanation(predicted_class, heatmap_path, original_img_path):
    try:
        import base64
        from groq import Groq

        client = Groq(api_key=GROQ_API_KEY)

        with open(original_img_path, "rb") as f:
            original_b64 = base64.b64encode(f.read()).decode("utf-8")

        with open(heatmap_path, "rb") as f:
            heatmap_b64 = base64.b64encode(f.read()).decode("utf-8")

        ext  = os.path.splitext(original_img_path)[1].lower()
        mime = "image/png" if ext == ".png" else "image/jpeg"

        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Image 1 — This is the original skin image uploaded by the patient:"
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{original_b64}"}
                        },
                        {
                            "type": "text",
                            "text": "Image 2 — This is the Grad-CAM explainability heatmap generated by the deep learning model:"
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{heatmap_b64}"}
                        },
                        {
                            "type": "text",
                            "text": f"""You are a medical AI assistant specializing in dermatology and explainable AI.

The deep learning model predicted: {predicted_class}

In the Grad-CAM heatmap:
- Red / Yellow / White regions = areas the model focused on MOST strongly
- Blue / Purple / Dark regions = areas the model considered LESS relevant

Please provide a clear structured explanation with these exact sections:

**1. What the Model Saw (Heatmap Analysis)**
Describe which regions of the skin image are highlighted in the heatmap and why those specific visual features such as texture, color, pattern, or lesion shape likely led the model to predict {predicted_class}.

**2. About {predicted_class}**
Brief medical description — what this condition is, who typically gets it, and how it appears on skin.

**3. Key Visual Signs Detected**
List the specific visual characteristics visible in the original image that match {predicted_class}, such as redness, blisters, scaling, pigmentation changes, border irregularity, and so on.

**4. Recommended Next Steps**
Practical advice — urgency level, type of specialist to consult, and any immediate precautions the patient should take.

Write in a clear, empathetic tone that a non-medical person can easily understand."""
                        }
                    ]
                }
            ],
            max_tokens=1024
        )

        print("✅ Groq explanation generated successfully")
        return response.choices[0].message.content

    except Exception as e:
        import traceback
        print("Groq API Error:", e)
        traceback.print_exc()
        return "AI explanation could not be generated at this time. Please consult a dermatologist."


# ============================================================
# Home Route
# ============================================================

@app.route("/", methods=["GET", "POST"])
def index():

    prediction   = None
    image_path   = None
    heatmap_path = None
    explanation  = None

    if request.method == "POST":

        if "file" not in request.files:
            return render_template("index1.html", prediction="No file uploaded")

        file = request.files["file"]

        if file.filename == "":
            return render_template("index1.html", prediction="No file selected")

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            prediction, pred_index, img_array = predict_image(filepath)
            image_path = filepath

            if pred_index is not None:
                print("✅ pred_index:", pred_index)
                heatmap_save_path = generate_gradcam(filepath, img_array, pred_index)
                print("✅ heatmap_save_path:", heatmap_save_path)

                if heatmap_save_path:
                    heatmap_path = "static/heatmaps/gradcam.jpg"
                    print("⏳ Calling Groq API...")
                    explanation = get_ai_explanation(
                        prediction,
                        heatmap_save_path,
                        filepath
                    )
                    print("✅ Explanation length:", len(explanation) if explanation else 0)
                else:
                    print("❌ heatmap_save_path is None — GradCAM failed")
            else:
                print("❌ pred_index is None — prediction failed")

    return render_template(
        "index1.html",
        prediction   = prediction,
        image_path   = image_path,
        heatmap_path = heatmap_path,
        explanation  = explanation
    )


# ============================================================
# Run Flask App
# ============================================================

if __name__ == "__main__":
    app.run(debug=True)