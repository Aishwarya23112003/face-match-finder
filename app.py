# face_match_env\Scripts\activate  --> activate environment 
# python app.py --> run the app
from flask import Flask, render_template, request
import os
import cv2
import shutil
from werkzeug.utils import secure_filename
from main import FaceMatcher

app = Flask(__name__, static_url_path='/static')
app.config['UPLOAD_FOLDER'] = 'uploads'
STATIC_MATCHED_DIR = os.path.join('static', 'matched')
os.makedirs(STATIC_MATCHED_DIR, exist_ok=True)

# Create one global instance of matcher
matcher = FaceMatcher()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', matches=None)

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('photo')
    match_filenames = []

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Read the uploaded image
        img = cv2.imread(filepath)
        os.remove(filepath)

        # Get embedding
        query_embedding = matcher.get_embedding(img)
        if query_embedding is not None:
            matches = matcher.find_similar_faces(query_embedding)

            # Clean previous matched images
            for f in os.listdir(STATIC_MATCHED_DIR):
                os.remove(os.path.join(STATIC_MATCHED_DIR, f))

            # Copy matched images into static/matched/
            for result_path in matches:
                fname = os.path.basename(result_path)
                shutil.copy(result_path, os.path.join(STATIC_MATCHED_DIR, fname))
            
            # Convert to static-relative path for rendering
            matches = [f'static/results/{os.path.basename(p)}' for p in matches]

    return render_template('index.html', matches=matches)

@app.route('/webcam', methods=['GET'])
def webcam():
    img = matcher.capture_face()
    matches = []

    if img is not None:
        query_embedding = matcher.get_embedding(img)
        if query_embedding is not None:
            matches = matcher.find_similar_faces(query_embedding)

            # Clean and copy matched images to static folder
            STATIC_MATCHED_DIR = os.path.join('static', 'results')
            os.makedirs(STATIC_MATCHED_DIR, exist_ok=True)

            # Clear old files
            for file in os.listdir(STATIC_MATCHED_DIR):
                os.remove(os.path.join(STATIC_MATCHED_DIR, file))

            # Copy new results to static folder
            for path in matches:
                fname = os.path.basename(path)
                shutil.copy(path, os.path.join(STATIC_MATCHED_DIR, fname))

            # Convert to static-relative path for rendering
            matches = [f'static/results/{os.path.basename(p)}' for p in matches]


    return render_template('index.html', matches=matches)


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(STATIC_MATCHED_DIR, exist_ok=True)
    app.run(debug=True)
