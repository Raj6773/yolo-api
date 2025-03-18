from flask import Flask, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
import base64

app = Flask(__name__)

# Enable CORS (Cross-Origin Resource Sharing)
from flask_cors import CORS
CORS(app)

# Load YOLOv8 model
model = YOLO("yolo-api/best.pt")  # NEW: Points to the correct location inside yolo-api folder

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get base64 image from request
        data = request.json
        img_data = base64.b64decode(data['image'])
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Run YOLO model on the image
        results = model(img)

        # Convert results to JSON format
        detections = []
        for r in results:
            for box in r.boxes:
                detections.append({
                    "class": int(box.cls),
                    "confidence": float(box.conf),
                    "bbox": box.xyxy.tolist()[0]  # Convert tensor to list
                })

        return jsonify({"detections": detections})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the API
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
