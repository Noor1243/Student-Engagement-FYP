from flask import Flask, render_template, request, send_from_directory, jsonify
import os
from main_processor import process_video

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Make sure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'video' not in request.files:
        return "No video part", 400
    file = request.files['video']
    if file.filename == '':
        return "No selected video", 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'my_video.mp4')
    file.save(filepath)

    # 🧠 Process video and get frames list
    frame_files = process_video(filepath)

    return jsonify({'frames': frame_files})

@app.route('/output/<path:filename>')
def output_files(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
