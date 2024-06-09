from flask import Flask, jsonify, request
from flask_cors import CORS
import cv2
import numpy as np
import torch
import base64
from yolov5 import YOLOv5

app = Flask(__name__)
CORS(app)

# Load YOLOv5 model
model = YOLOv5('yolov5s.pt')  # replace with the path to your model

@app.route('/process_frame', methods=['POST'])
def process_frame():
    data = request.json
    image_data = data['image']

    # Decode image
    try:
        image_data = base64.b64decode(image_data.split(',')[1])
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify(error=f"Error decoding image: {e}"), 400

    if frame is None or frame.size == 0:
        return jsonify(error="Empty frame"), 400

    # Process the frame with YOLOv5
    results = model.predict(frame)

    # Extract detections
    detections = []
    for result in results.xyxy[0].cpu().numpy():
        x1, y1, x2, y2, conf, cls = result
        detections.append({
            'x1': int(x1),
            'y1': int(y1),
            'x2': int(x2),
            'y2': int(y2),
            'confidence': float(conf),
            'class': int(cls)
        })

    return jsonify(success=bool(detections), detections=detections)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

