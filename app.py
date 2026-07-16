import os
import time
import numpy as np

from flask import Flask, render_template, request
from PIL import Image

import torch
import torch.nn.functional as F

from model import create_model

# ==========================================
# Flask App
# ==========================================
app = Flask(__name__)

# ==========================================
# Device
# ==========================================
device = torch.device("cpu")

# ==========================================
# Load trained model
# ==========================================
print("Loading model...", flush=True)

checkpoint = torch.load("astro_model_v2.pth", map_location=device)

classes = checkpoint["classes"]
num_classes = checkpoint["num_classes"]

model = create_model(num_classes)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

print("✅ AI model loaded successfully!", flush=True)
print("Torch version:", torch.__version__, flush=True)

print("Testing torch...", flush=True)

start = time.time()

x = torch.rand((3, 224, 224))

print("Created random tensor", flush=True)

x = x.float()

print("Converted to float", flush=True)

x = x / 255.0

print("Division OK", flush=True)

print(f"Test took {time.time()-start:.3f}s", flush=True)

# ==========================================
# Normalization constants
# ==========================================
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# ==========================================
# Home page
# ==========================================
@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence = None
    error = None

    if request.method == "POST":

        print("🔥 POST STARTED", flush=True)

        try:

            files = request.files
            file = files.get("image")

            if file is None:
                return render_template(
                    "index.html",
                    error="No image uploaded"
                )

            print(file.filename, flush=True)

            # -----------------------------
            # Open Image
            # -----------------------------
            start = time.time()

            image = Image.open(file).convert("RGB")

            print(f"Image opened ({time.time()-start:.2f}s)", flush=True)
            print("Original size:", image.size, flush=True)

            # -----------------------------
            # Resize
            # -----------------------------
            print("Resize...", flush=True)

            image = image.resize((224, 224))

            print("Resize done", flush=True)

            # -----------------------------
            # Convert to NumPy
            # -----------------------------
            print("Converting to numpy...", flush=True)

            arr = np.asarray(image, dtype=np.float32)

            print("NumPy OK", flush=True)

            # -----------------------------
            # Convert to Tensor
            # -----------------------------
            print("Creating tensor...", flush=True)

            image = torch.from_numpy(arr)

            print("Tensor OK", flush=True)

            # -----------------------------
            # CHW
            # -----------------------------
            print("Permuting...", flush=True)

            image = image.permute(2, 0, 1)

            print("Permute OK", flush=True)

            # -----------------------------
            # Scale
            # -----------------------------
            print("Scaling...", flush=True)

            image = image / 255.0

            print("Scale OK", flush=True)

            # -----------------------------
            # Normalize
            # -----------------------------
            print("Normalizing...", flush=True)

            image = (image - mean) / std

            print("Normalize OK", flush=True)

            image = image.unsqueeze(0).to(device)

            print("Tensor ready", flush=True)

            # -----------------------------
            # Model
            # -----------------------------
            print("Running model...", flush=True)

            start = time.time()

            with torch.no_grad():
                outputs = model(image)

            print(f"Model finished ({time.time()-start:.2f}s)", flush=True)

            outputs = torch.flatten(outputs, start_dim=1)

            probabilities = F.softmax(outputs, dim=1)

            confidence_tensor, predicted = torch.max(probabilities, 1)

            prediction = classes[predicted.item()]
            confidence = round(confidence_tensor.item() * 100, 2)

            print(f"Prediction: {prediction}", flush=True)
            print(f"Confidence: {confidence}%", flush=True)

        except Exception as e:

            import traceback

            print("ERROR:", e, flush=True)
            traceback.print_exc()

            error = str(e)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        error=error
    )


# ==========================================
# Run Website
# ==========================================
if __name__ == "__main__":

    print("Starting Flask...", flush=True)

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port
    )
