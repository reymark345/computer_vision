from flask import Flask, request, jsonify
import base64
import os
from ultralytics import YOLO
from datetime import datetime

app = Flask(__name__)

# Create uploads and results folders if not exists
UPLOAD_FOLDER = 'server/uploads'
RESULTS_FOLDER = 'server/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Initialize YOLO model once at startup
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_path = os.path.join(workspace_root, "runs/detect/train/weights/best.pt")
model = YOLO(model_path)
print(f"\n✓ YOLO model loaded from: {model_path}")

@app.route('/api/upload', methods=['POST'])
def upload_image():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data received'}), 400
        
        image_id = data.get('id')
        image_base64 = data.get('image')
        created_at = data.get('created_at')
        
        print(f"\n{'='*50}")
        print(f"Received image upload request:")
        print(f"  ID: {image_id}")
        print(f"  Created at: {created_at}")
        print(f"  Image size: {len(image_base64) if image_base64 else 0} chars (base64)")
        
        if not image_base64:
            return jsonify({'error': 'No image data'}), 400
        
        # Decode Base64 image
        image_bytes = base64.b64decode(image_base64)
        
        # Save image to file
        filename = f"mango_{image_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        with open(filepath, 'wb') as f:
            f.write(image_bytes)
        
        print(f"  Saved to: {filepath}")
        print(f"  File size: {len(image_bytes)} bytes")
        
        # Run YOLO prediction on the uploaded image
        print(f"\n  Running YOLO detection...")
        results = model.predict(
            source=filepath,
            conf=0.25,
            save=True,
            project=os.path.dirname(os.path.abspath(RESULTS_FOLDER)),
            name=os.path.basename(os.path.abspath(RESULTS_FOLDER)),
            exist_ok=True
        )
        
        print(f"  ✓ Detection completed")
        print(f"  Results saved to: {RESULTS_FOLDER}/")
        print(f"{'='*50}\n")
        
        return jsonify({
            'success': True,
            'message': 'Image uploaded and processed successfully',
            'filename': filename,
            'detections': len(results[0].boxes) if results and len(results) > 0 else 0
        }), 200
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok'}), 200

if __name__ == '__main__':
    print("\n" + "="*50)
    print("Flask Test Server for Mango Image Upload")
    print("="*50)
    # print(f"Upload endpoint: http://172.31.246.40:5000/api/upload")
    # print(f"Health check:    http://172.31.246.40:5000/health")
    print(f"Upload endpoint: http://192.168.254.108:5000/api/upload")
    print(f"Health check:    http://192.168.254.108:5000/health")

    
    print(f"Images saved to: ./server/{UPLOAD_FOLDER}/")
    print(f"Results saved to: ./server/{RESULTS_FOLDER}/")
    print("="*50 + "\n")
    
    # Run on 0.0.0.0 to accept connections from other devices
    app.run(host='0.0.0.0', port=5000, debug=True)

