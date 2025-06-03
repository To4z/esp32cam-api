from flask import Flask, request, jsonify
import numpy as np
import cv2
from lp_image import detect_plates_from_image

app = Flask(__name__)

@app.route('/')
def home():
    return '🚗 API Nhận Diện Biển Số Đang Hoạt Động!'

@app.route('/detect-plate', methods=['POST'])
def detect_plate():
    if 'image' not in request.files:
        return jsonify({'error': 'Thiếu ảnh'}), 400

    file = request.files['image']
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    plates = detect_plates_from_image(img)
    return jsonify({'plates': plates})
import os

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port)

