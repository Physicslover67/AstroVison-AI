import os
import time
import psutil
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
# RAM Monitor
# ==========================================
process = psutil.Process(os.getpid())

def ram(stage):
    print(
        f"[{stage}] RAM = {process.memory_info().rss / 1024 / 1024:.1f} MB",
        flush=True
    )

# ==========================================
# Load trained model
# ==========================================
print("Loading model...", flush=True)
ram("Before loading model")

checkpoint = torch.load("astro_model_v2.pth", map_location=device)

ram("After checkpoint")

classes = checkpoint["classes"]
num_classes = checkpoint["num_classes"]

model = create_model(num_classes)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

ram("After model")

print("✅ AI model loaded successfully!", flush=True)
print("Torch version:", torch.__version__, flush=True)

# ==========================================
# Normalization constants
# ==========================================
mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
std = torch.tensor([0.229,0.224,0.225]).view(3,1,1)

ram("Startup complete")

# ==========================================
# Home Page
# ==========================================
@app.route("/", methods=["GET","POST"])
def home():

    prediction = None
    confidence = None
    error = None

    if request.method == "POST":

        print("🔥 POST STARTED", flush=True)
        ram("Request Start")

        try:

            file = request.files.get("image")

            if file is None:
                return render_template(
                    "index.html",
                    error="No image uploaded"
                )

            print(file.filename, flush=True)

            # -----------------------------
            # Open image
            # -----------------------------
            start = time.time()

            image = Image.open(file).convert("RGB")

            print(f"Image opened ({time.time()-start:.2f}s)", flush=True)
            print("Original size:", image.size, flush=True)

            ram("After Open")

            # -----------------------------
            # Resize
            # -----------------------------
            print("Resize...", flush=True)

            image = image.resize((224,224))

            print("Resize done", flush=True)

            ram("After Resize")

            # -----------------------------
            # NumPy
            # -----------------------------
            print("Converting to numpy...", flush=True)

            arr = np.asarray(image,dtype=np.float32)

            print("NumPy OK", flush=True)

            ram("After NumPy")

            # -----------------------------
            # Tensor
            # -----------------------------
            print("Creating tensor...", flush=True)

            image = torch.from_numpy(arr)

            print("Tensor OK", flush=True)

            ram("After Tensor")

            # -----------------------------
            # Permute
            # -----------------------------
            arr = np.transpose(arr, (2,0,1)).copy()

            print("NumPy transpose OK", flush=True)

            image = torch.from_numpy(arr)

            print("Torch tensor OK", flush=True)

            # -----------------------------
            # Float
            # -----------------------------
            print("Converting to float...", flush=True)

            start = time.time()

            image = image.float()

            print(f"float() OK ({time.time()-start:.4f}s)", flush=True)

            ram("After Float")

            # -----------------------------
            # Scale
            # -----------------------------
            print("Before clone", flush=True)
            image = image.clone()
            print("After clone", flush=True)

            print("Before mul_", flush=True)
            image.mul_(1.0 / 255.0)
            print("After mul_", flush=True)
            # -----------------------------
            # Normalize
            # -----------------------------
            print("Normalizing...", flush=True)

            image = (image-mean)/std

            print("Normalize OK", flush=True)

            ram("After Normalize")

            image = image.unsqueeze(0).to(device)

            print("Tensor ready", flush=True)

            ram("After Unsqueeze")

            # -----------------------------
            # Model
            # -----------------------------
            print("Running model...", flush=True)

            start = time.time()

            with torch.no_grad():
                outputs = model(image)

            print(f"Model finished ({time.time()-start:.2f}s)", flush=True)

            ram("After Model")

            outputs = torch.flatten(outputs,start_dim=1)

            probabilities = F.softmax(outputs,dim=1)

            confidence_tensor,predicted = torch.max(probabilities,1)

            prediction = classes[predicted.item()]
            confidence = round(confidence_tensor.item()*100,2)

            print("Prediction:",prediction,flush=True)
            print("Confidence:",confidence,flush=True)

        except Exception as e:

            import traceback

            print("ERROR:",e,flush=True)
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

    port = int(os.environ.get("PORT",5000))

    app.run(
        host="0.0.0.0",
        port=port
    )

