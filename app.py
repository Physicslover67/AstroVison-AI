import os
import time
from flask import Flask, render_template, request
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
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

# ==========================================
# Image preprocessing
# ==========================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

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

            print("A", flush=True)

            files = request.files

            print("B", flush=True)
            print(files.keys(), flush=True)

            print("C", flush=True)

            file = files.get("image")

            print("D", flush=True)

            if file is None:
                print("❌ No image uploaded", flush=True)
                return render_template(
                    "index.html",
                    prediction=None,
                    confidence=None,
                    error="No image uploaded"
                )

            print("E", flush=True)
            print(file.filename, flush=True)

            # -----------------------------
            # Open image
            # -----------------------------
            start = time.time()

            image = Image.open(file).convert("RGB")

            print(f"2. Image opened ({time.time()-start:.2f}s)", flush=True)

            # -----------------------------
            # Transform
            # -----------------------------
            print("Before transform", flush=True)

            image = transform(image)

            print("After transform", flush=True)

            image = image.unsqueeze(0)

            print("After unsqueeze", flush=True)

            image = image.to(device)

            print("After to(device)", flush=True)

            tensor_image = image

            # -----------------------------
            # Model
            # -----------------------------
            print("Before model()", flush=True)

            start = time.time()

            with torch.no_grad():
                outputs = model(tensor_image)

            print(f"After model() ({time.time()-start:.2f}s)", flush=True)

            # -----------------------------
            # Softmax
            # -----------------------------
            outputs = torch.flatten(outputs, start_dim=1)

            probabilities = F.softmax(outputs, dim=1)

            confidence_tensor, predicted = torch.max(probabilities, 1)

            prediction = classes[predicted.item()]
            confidence = round(confidence_tensor.item() * 100, 2)

            print(f"Prediction: {prediction}", flush=True)
            print(f"Confidence: {confidence}%", flush=True)

        except Exception as e:

            print("❌ ERROR:", e, flush=True)

            import traceback
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

    print("🚀 Starting Flask server...", flush=True)

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port
    )
