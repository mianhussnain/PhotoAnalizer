import os
from flask import Flask, request, send_file, jsonify
import cv2
import numpy as np
from rembg import remove, new_session
from PIL import Image
from matplotlib import pyplot as plt 
from matplotlib.patches import FancyArrowPatch , Rectangle
from matplotlib.patches import FancyArrowPatch, Rectangle
from io import BytesIO

app = Flask(__name__)

plt.rcParams['figure.figsize'] = (10.0, 8.0)

def initialize_session(model_name):
    return new_session(model_name)

def process_image(session, image, size=None, bgcolor='white'):
    if size is not None:
        image = image.resize(size)
    else:
        size = image.size
    result = Image.new("RGB", size, bgcolor)
    out = remove(image, session=session)
    result.paste(out, mask=out)
    return result

def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('./cv2_cascade_classifier/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    return faces

def draw_guidelines_on_faces(image, faces):
    for (x, y, w, h) in faces:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.axis('off')

        linewidthh = 1.5
        rect = Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax.axhline(y, color=(203/255, 12/255, 255/255), linewidth=linewidthh)
        ax.axhline(y + h, color=(203/255, 12/255, 255/255), linewidth=linewidthh)
        ax.axvline(image.shape[1] - 50, color=(203/255, 12/255, 255/255), linewidth=linewidthh)
        ax.axvline(10, color=(203/255, 12/255, 255/255), linewidth=linewidthh)
        ax.axhline(image.shape[0]*0, color=(203/255, 12/255, 255/255), linewidth=linewidthh)
        ax.axhline(image.shape[0], color=(203/255, 12/255, 255/255), linewidth=linewidthh)
        vertical_line_x = image.shape[1] - 1.5
        arrow1 = FancyArrowPatch((vertical_line_x, y), (vertical_line_x, y + h),
                                 color=(203/255, 12/255, 255/255), linewidth=linewidthh,
                                 arrowstyle='-|>', mutation_scale=10, zorder=5)
        arrow2 = FancyArrowPatch((vertical_line_x, y + h), (vertical_line_x, y),
                                 color=(203/255, 12/255, 255/255), linewidth=linewidthh,
                                 arrowstyle='-|>', mutation_scale=10, zorder=5)

        vertical_line_height_percentage = (h / image.shape[0]) * 100
        ax.text(vertical_line_x - 10, (y + y + h) // 2, f"{vertical_line_height_percentage:.2f}%",
                color=(0, 0, 0), fontsize=8, rotation=90)
        ax.add_patch(arrow1)
        ax.add_patch(arrow2)

        # Set the figure size explicitly
        fig.set_size_inches(image.shape[1] / 100, image.shape[0] / 100)

        # Save the figure to BytesIO
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close()

        return buf

@app.route('/process_image', methods=['POST'])
def process_uploaded_image():
    # Check if the POST request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    # If the user does not select a file, the browser submits an empty file without a filename
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Ensure the file is an allowed type
    allowed_extensions = {'png', 'jpg', 'jpeg'}
    if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
        return jsonify({'error': 'Invalid file type'})

    try:
        # Read the uploaded image
        uploaded_image = Image.open(file)
    except Exception as e:
        return jsonify({'error': f'Error reading the file: {str(e)}'})

    # Process the image
    session = initialize_session("u2net_human_seg")
    processed_image = process_image(session, uploaded_image, size=(600, 600), bgcolor='#FFFFFF')

    # Save the processed image to BytesIO
    output_buf = BytesIO()
    processed_image.save(output_buf, format='png')
    output_buf.seek(0)

    # Detect faces and draw guidelines
    faces = detect_faces(np.array(processed_image))
    guidelines_buf = draw_guidelines_on_faces(np.array(processed_image), faces)

    return send_file(output_buf, mimetype='image/png', as_attachment=True, download_name='processed_image.png', headers={'Cache-Control': 'no-cache, no-store, must-revalidate'},
                     attachment_filename='processed_image.png'), send_file(guidelines_buf, mimetype='image/png', as_attachment=True,
                                                                      download_name='guidelines.png', headers={'Cache-Control': 'no-cache, no-store, must-revalidate'},
                                                                      attachment_filename='guidelines.png')

if __name__ == "__main__":
    app.run(debug=True)
