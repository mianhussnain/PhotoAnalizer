import os
from flask import Flask, request, jsonify
import cv2
import numpy as np
from rembg import remove, new_session
from PIL import Image
import datetime
import matplotlib
from matplotlib import pyplot as plt 
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle
matplotlib.use("Agg")
plt.rcParams['figure.figsize'] = (10.0, 10.0)

photoAnalyzer = Flask(__name__)
face_cascade = cv2.CascadeClassifier('./cv2_cascade_classifier/haarcascade_frontalface_default.xml')

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

def detect_faces(test_image):
    gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(200, 200))
    shoulders_roi = test_image

    for (x, y, w, h) in faces:
        margin = 150
        x_roi = max(0, x - margin)
        y_roi = max(0, y - margin)
        w_roi = min(test_image.shape[1], w + 2 * margin)
        h_roi = min(test_image.shape[0], h + 2 * margin)
        shoulders_roi = test_image[y_roi:y_roi + h_roi, x_roi:x_roi + w_roi]

        gray = cv2.cvtColor(shoulders_roi, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(200, 200))

    return shoulders_roi, faces

def is_red_eye(eye_image):
    eye_hsv = cv2.cvtColor(eye_image, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 100, 100], dtype=np.uint8)
    upper_red = np.array([10, 255, 255], dtype=np.uint8)
    red_mask = cv2.inRange(eye_hsv, lower_red, upper_red)
    red_pixel_percentage = np.count_nonzero(red_mask) / (eye_image.shape[0] * eye_image.shape[1]) * 100
    
    red_eye_threshold = 80.0  
    return red_pixel_percentage > red_eye_threshold


def add_padding_background(image, faces, output_path):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.axis('off')
    fig.set_size_inches((image.shape[1] / 100), (image.shape[0] / 100) + 2)

    plt.savefig(output_path, bbox_inches="tight", pad_inches=1)
    plt.close()

    test_image = cv2.imread(output_path)
    gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(200, 200))

    is_red_eye_detected = draw_guidelines_on_faces(test_image, faces, output_path)
    return is_red_eye_detected, faces


def draw_guidelines_on_faces(image, faces, output_path):
    for (x, y, w, h) in faces:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.axis('off')
        redeye =False

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        eye_cascade = cv2.CascadeClassifier('./cv2_cascade_classifier/haarcascade_eye.xml')
        roi_gray = gray[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5, minSize=(30, 30))

        for (ex, ey, ew, eh) in eyes:
            eye_image = image[y + ey:y + ey + eh, x + ex:x + ex + ew]

            if is_red_eye(eye_image):
                redeye =True
                print("Red-eye detected!")
            else:
                redeye =False
                print("No red-eye detected.")

        linewidthh = 1.5
        ax.axhline(y, color=(30/255, 30/255, 31/255), linewidth=linewidthh)
        ax.axhline(y + h, color=(30/255, 30/255, 31/255), linewidth=linewidthh)
        ax.axvline(min(image.shape[1], w +2 * 150), color=(255/255, 31/255, 31/255), linewidth=linewidthh)
        ax.axvline(max(0, x - 100), color=(255/255, 31/255, 31/255), linewidth=linewidthh)
        ax.axhline(max(0, y - 90), color=(255/255, 31/255, 31/255), linewidth=linewidthh)
        ax.axhline(min(image.shape[0], h + 2 * 150), color=(255/255, 31/255, 31/255), linewidth=linewidthh)
        
        vertical_line_x = image.shape[1] - 8
        arrow1 = FancyArrowPatch((vertical_line_x, y), (vertical_line_x, y + h),
                                 color=(30/255, 30/255, 31/255), linewidth=linewidthh,
                                 arrowstyle='-|>', mutation_scale=10, zorder=5)
        arrow2 = FancyArrowPatch((vertical_line_x, y + h), (vertical_line_x, y),
                                 color=(30/255, 30/255, 31/255), linewidth=linewidthh,
                                 arrowstyle='-|>', mutation_scale=10, zorder=5)
        
        arrow3 = FancyArrowPatch((image.shape[1] - 50, max(0, y - 90)), (image.shape[1] - 50, min(image.shape[0], h + 2 * 150)),
                                 color=(255/255, 31/255, 31/255), linewidth=linewidthh,
                                 arrowstyle='-|>', mutation_scale=10, zorder=5)
        arrow4 = FancyArrowPatch((image.shape[1] - 50, max(0, y - 90)), (image.shape[1] - 50, min(image.shape[0], h + 2 * 150)),
                                 color=(255/255, 31/255, 31/255), linewidth=linewidthh,
                                 arrowstyle='<|-', mutation_scale=10, zorder=5)
        
        arrow5 = FancyArrowPatch((max(0, x - 100), y-130), (min(image.shape[1], w + 2 * 150),  y-130),
                         color=(255/255, 31/255, 31/255), linewidth=linewidthh,
                         arrowstyle='-|>', mutation_scale=10, zorder=5)
        arrow6 = FancyArrowPatch((max(0, x - 100), y-130), (min(image.shape[1], w + 2 * 150),  y-130),
                         color=(255/255, 31/255, 31/255), linewidth=linewidthh,
                         arrowstyle='<|-', mutation_scale=10, zorder=5)



        vertical_line_height_percentage = (h / image.shape[0]) * 100
        ax.text(vertical_line_x - 20, (y + y + h) // 2, f"{vertical_line_height_percentage:.2f}%",
                color=(30/255, 30/255, 31/255), fontsize=16, rotation=90)
        
        ax.text(vertical_line_x - 70, (y + y + h) // 2, '600',
                color=(255/255, 31/255, 31/255), fontsize=16, rotation=90)
        
        ax.text(image.shape[1]/2, y-140, '600',
                color=(255/255, 31/255, 31/255), fontsize=16)

        ax.add_patch(arrow1)
        ax.add_patch(arrow2)
        ax.add_patch(arrow3)
        ax.add_patch(arrow4)
        ax.add_patch(arrow6)
        ax.add_patch(arrow5)


        plt.savefig(output_path, bbox_inches="tight", pad_inches=0.3)
        plt.close()
        return redeye

@photoAnalyzer.route('/photoAnalyzer', methods=['POST'])
def process_directory():
    session = initialize_session("u2net_human_seg")
    draw_guidelines = True

    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({"error": "Empty image provided"}), 400

    if not file.filename.endswith(('.jpg', '.jpeg', '.png')):
        return jsonify({"error": "Invalid file extension"}), 400

    now = datetime.datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    image_name = "upload" + timestamp_str + ".jpg"

    tmp_directory = 'temp_directory'
    if not os.path.exists(tmp_directory):
        os.makedirs(tmp_directory)

    image_path = os.path.join(tmp_directory, image_name)

    output_directory = "./output_images_directory"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    output_path = os.path.join(output_directory, f"output_{image_name}")

    is_face_detected = False
    is_red_eye_detected = False
    is_face_one = True

    try:
        file.save(image_path)
        with Image.open(image_path) as image:
            processed_image = process_image(session, image, size=(600, 600), bgcolor='#FFFFFF')
            processed_image.save(output_path)
            test_image, detected_faces = detect_faces(cv2.imread(output_path))

        if len(detected_faces) == 0:
            print(f'Face not detected in {image_name}')
            return jsonify({"result": {"isFaceDetected": False}})

        is_face_detected = True
        white_background = Image.new("RGB", (600, 600), "white")
        white_background.paste(Image.fromarray(test_image), (0, 0))
        white_background.save(output_path)

        if len(detected_faces) > 1:
            is_face_one = False
            return jsonify({"result": {'isFaceDetected': is_face_detected, 'output_image_path': None, 'isFaceOne': is_face_one}})
        elif len(detected_faces) == 1:
            is_face_one = True

        if draw_guidelines:
            for face in detected_faces:
                is_red_eye_detected, facese = add_padding_background(test_image, [face], output_path)
                if len(facese) == 0:
                    is_face_detected = False
                    print(f'Face not detected in {image_name}')
                    return jsonify({"result": {"isFaceDetected": is_face_detected}})
                if len(facese) > 1:
                    is_face_one = False
                    return jsonify({"result": {'isRedEye': is_red_eye_detected, 'isFaceDetected': is_face_detected,
                                                'output_image_path': None, 'isFaceOne': is_face_one}})
                elif len(facese) == 1:
                    is_face_one = True

    except Exception as e:
        print(f"Error processing image: {str(e)}")

    result = {
        'isRedEye': is_red_eye_detected,
        'isFaceDetected': is_face_detected,
        'output_image_path': output_path,
        'isFaceOne': is_face_one
    }

    return jsonify({"result": result})

if __name__ == "__main__":
    photoAnalyzer.run(debug=True, host='0.0.0.0', threaded=True)
