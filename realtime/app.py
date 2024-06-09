import os
import logging
import json
import base64
import cv2
from pathlib import Path
from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
from ultralytics import YOLO
import random
import shutil

app = Flask(__name__)
#CORS(app)
CORS(app, resources={r"/*": {"origins": "*"}})
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load YOLOv8 model (ensure the model path is correct)
default_model = YOLO('yolov8n.pt')  # Replace 'yolov8n.pt' with your default model path

# COCO class labels
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'TV', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]



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
        app.logger.error(f"Error decoding image: {e}")
        return jsonify(error="Error decoding image"), 400

    if frame is None or frame.size == 0:
        app.logger.error("Empty frame received")
        return jsonify(error="Empty frame"), 400

    # Process the frame with YOLOv8
    results = default_model(frame)
    
    # Extract person detections
    detections = []
    for result in results[0].boxes:
        x1, y1, x2, y2 = result.xyxy[0]
        confidence = result.conf[0]
        cls = int(result.cls)
        class_name = COCO_CLASSES[cls]
        detections.append({
            'x1': int(x1),
            'y1': int(y1),
            'x2': int(x2),
            'y2': int(y2),
            'confidence': float(confidence),
            'class': class_name
        })

    return jsonify(success=bool(detections), detections=detections) if detections else jsonify(success=False)

@app.route('/save-annotations', methods=['POST'])
def save_annotations():
    try:
        data = request.json
        logger.info(f"Starting Annotation with data: {data}")

        annotations = data['annotations']
        dataset_folder = data['datasetFolder']
        train_percent = data['trainPercent']
        val_percent = data['valPercent']
        test_percent = data['testPercent']
        tags = data['tags']

        base_dir = os.path.join('/home/ubuntu/dev/realtime', dataset_folder)
        os.makedirs(os.path.join(base_dir, 'train', 'images'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, 'train', 'labels'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, 'valid', 'images'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, 'valid', 'labels'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, 'test', 'images'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, 'test', 'labels'), exist_ok=True)

        # Shuffle annotations
        random.shuffle(annotations)

        # Calculate split indices
        train_end = int(len(annotations) * (train_percent / 100))
        val_end = train_end + int(len(annotations) * (val_percent / 100))

        train = annotations[:train_end]
        val = annotations[train_end:val_end]
        test = annotations[val_end:]

        def save_annotations(data, type):
            for anno in data:
                image_id = anno['imageId']
                label = anno['label']
                try:
                    label_index = tags.index(label)
                except ValueError:
                    continue
                x_center = anno['x_center'] / 100
                y_center = anno['y_center'] / 100
                bbox_width = anno['bbox_width'] / 100
                bbox_height = anno['bbox_height'] / 100
                image_data = anno['imageData']

                image_filename = os.path.basename(image_id) + '.png'
                label_filename = os.path.basename(image_id) + '.txt'

                label_file = os.path.join(base_dir, type, 'labels', label_filename)
                with open(label_file, 'a') as f:
                    annotation_str = f"{label_index} {x_center} {y_center} {bbox_width} {bbox_height}\n"
                    f.write(annotation_str)

                image_file = os.path.join(base_dir, type, 'images', image_filename)
                with open(image_file, 'wb') as f:
                    try:
                        if image_data.startswith('data:image'):
                            f.write(base64.b64decode(image_data.split(',')[1]))
                    except Exception as e:
                        continue

        save_annotations(train, 'train')
        save_annotations(val, 'valid')
        save_annotations(test, 'test')

        data_yaml = f"""
        train: {os.path.join(base_dir, 'train', 'images')}
        val: {os.path.join(base_dir, 'valid', 'images')}
        nc: {len(tags)}
        names: {json.dumps(tags)}
        """

        data_yaml_path = os.path.join(base_dir, 'data.yaml')
        with open(data_yaml_path, 'w') as f:
            f.write(data_yaml)

        return jsonify(message='Annotations saved and training data prepared successfully.', data_yaml_path=data_yaml_path)
    except Exception as e:
        return jsonify(error=f'Failed to parse request data. {str(e)}'), 500


@app.route('/start-training', methods=['GET'])
def start_training():
    dataset_folder = request.args.get('dataset_folder')
    epochs = request.args.get('epochs', default=100, type=int)
    batch_size = request.args.get('batch_size', default=16, type=int)
    save_best_model_name = request.args.get('save_best_model_path')  # This is the name of the new model file
    dataset_path = os.path.join('/home/ubuntu/dev/realtime', dataset_folder)
    data_yaml_path = os.path.join(dataset_path, 'data.yaml')
    train_dir = os.path.join('/home/ubuntu/dev/realtime/runs/detect', 'train')

    try:
        if not os.path.exists(data_yaml_path):
            raise FileNotFoundError(f"data.yaml not found at path: {data_yaml_path}")

        # Load the YOLO model
        model = YOLO('yolov8n.pt')  # Replace 'yolov8n.pt' with the path to your model if different
        
        # Remove existing 'train' directory if it exists
        if os.path.exists(train_dir):
            shutil.rmtree(train_dir)

        # Train the model
        model.train(data=data_yaml_path, epochs=epochs, batch=batch_size, imgsz=640, save_dir=train_dir)
        
        # Check if best.pt or last.pt exists and rename appropriately
        weights_dir = os.path.join(train_dir, 'weights')
        best_model_path = os.path.join(weights_dir, 'best.pt')
        last_model_path = os.path.join(weights_dir, 'last.pt')
        if save_best_model_name:
            save_best_model_path = os.path.join('/home/ubuntu/dev/realtime/models', save_best_model_name + '.pt')
            os.makedirs(os.path.dirname(save_best_model_path), exist_ok=True)
            
            if os.path.exists(best_model_path):
                os.rename(best_model_path, save_best_model_path)
            elif os.path.exists(last_model_path):
                os.rename(last_model_path, save_best_model_path)
            else:
                raise FileNotFoundError(f"No best.pt or last.pt found in {weights_dir}")

            return jsonify(message='Training completed successfully. Model saved.', best_model_path=save_best_model_path)
        else:
            return jsonify(message='Training completed successfully, but model name not provided.', best_model_path=None)

    except Exception as e:
        logger.error(f"Error starting training: {e}")
        return jsonify(error=f"Error starting training: {str(e)}"), 500
@app.route('/list-models', methods=['GET'])
def list_models():
    try:
        models_path = Path('/home/ubuntu/dev/realtime/models')
        models = [str(model_path) for model_path in models_path.glob('*.pt')]
        
        # Ensure 'yolov8n.pt' is always included
        if 'yolov8n.pt' not in models:
            models.append('yolov8n.pt')
        
        return jsonify(models=models)
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        return jsonify(error="Error listing models"), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        model_path = request.form['model']
        image_file = request.files['image']

        # Load the model
        model = YOLO(model_path)

        # Load the class labels
        if model_path == 'yolov8n.pt':
            class_labels = COCO_CLASSES
        else:
            data_yaml_path = Path(model_path).parent.parent / 'data.yaml'
            if data_yaml_path.exists():
                with open(data_yaml_path, 'r') as f:
                    data_yaml = yaml.safe_load(f)
                    class_labels = data_yaml['names']
            else:
                class_labels = COCO_CLASSES  # Fallback to COCO classes

        # Read image file
        image_data = image_file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Process the frame with YOLOv8
        results = model(frame)
        
        # Extract detections
        detections = []
        for result in results[0].boxes:
            x1, y1, x2, y2 = result.xyxy[0]
            confidence = result.conf[0]
            cls = int(result.cls)
            class_name = class_labels[cls]
            detections.append({
                'x1': int(x1),
                'y1': int(y1),
                'x2': int(x2),
                'y2': int(y2),
                'confidence': float(confidence),
                'class': class_name
            })

        if not detections:
            return jsonify(detections=[], message="No detections found.")

        return jsonify(detections=detections)
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        return jsonify(error="Error making prediction"), 500

@app.route('/list-train-images', methods=['GET'])
def list_train_images():
    try:
        train_images_path = Path('/home/ubuntu/dev/realtime/runs/detect/train')
        images = [str(image_path.name) for image_path in train_images_path.glob('*.png')]
        image_data = []
        for image_name in images:
            image_path = train_images_path / image_name
            with open(image_path, "rb") as img_file:
                image_base64 = base64.b64encode(img_file.read()).decode('utf-8')
                image_data.append({
                    "name": image_name,
                    "data": image_base64
                })
        return jsonify(images=image_data)
    except Exception as e:
        logger.error(f"Error retrieving train images: {e}")
        return jsonify(error="Error retrieving train images"), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
