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
checkpoint = torch.load("astro_model_v2.pth", map_location=device) 
classes = checkpoint["classes"] 
num_classes = checkpoint["num_classes"] 

model = create_model(num_classes) 
model.load_state_dict(checkpoint["model_state_dict"]) 
model.to(device) 
model.eval() 
print("✅ AI model loaded successfully!") 

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
    
    if request.method == "POST":
    print("🔥 POST STARTED", flush=True) 
        
        # Safe check to see if the file exists in the request
        if "image" not in request.files or request.files["image"].filename == "":
            return render_template("index.html", error="No image uploaded")
            
        file = request.files["image"] 
        
        print("🧠 Running AI Benchmarks...") 
        
        # 1. Open the image safely once
        start = time.time() 
        image = Image.open(file).convert("RGB") 
        print(f"2. Image opened ({time.time()-start:.2f}s)") 
        
        # 2. Transform the image
        start = time.time() 
        tensor_image = transform(image).unsqueeze(0).to(device) 
        print(f"3. Image transformed ({time.time()-start:.2f}s)") 
        
        # 3. Model Inference
        start = time.time() 
        print("4. Starting model...") 
        with torch.no_grad(): 
            outputs = model(tensor_image) 
        print(f"5. Model finished ({time.time()-start:.2f}s)") 
        
        # Post-processing
        outputs = torch.flatten(outputs, start_dim=1) 
        probabilities = F.softmax(outputs, dim=1) 
        confidence_tensor, predicted = torch.max(probabilities, 1) 
        
        prediction = classes[predicted.item()] 
        confidence = round(confidence_tensor.item() * 100, 2) 
        
        print(f"Prediction: {prediction}") 
        print(f"Confidence: {confidence}%") 
        
    return render_template( 
        "index.html", 
        prediction=prediction, 
        confidence=confidence 
    ) 

# ========================================== 
# Run Website 
# ========================================== 
if __name__ == "__main__": 
    print("🚀 Starting Flask server...") 
    port = int(os.environ.get("PORT", 5000)) 
    app.run(host="0.0.0.0", port=port)
