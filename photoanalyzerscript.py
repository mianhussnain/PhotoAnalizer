import os
import cv2
import numpy as np
from rembg import remove, new_session
from PIL import Image
from matplotlib import pyplot as plt 
from matplotlib.patches import FancyArrowPatch , Rectangle

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

def detect_faces(image_path):
    test_image = cv2.imread(image_path)
    gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('./cv2_cascade_classifier/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    return test_image, faces

def draw_guidelines_on_faces(image, faces, output_path):
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

        plt.savefig(output_path, bbox_inches='tight')
        plt.close()


def process_directory(input_directory, output_directory, draw_guidelines=True):
    session = initialize_session("u2net_human_seg")

    # Ensure the output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filename in os.listdir(input_directory):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_directory, filename)
            output_path = os.path.join(output_directory, f"output_{filename}")

            with Image.open(input_path) as image:
                processed_image = process_image(session, image, size=(600, 600), bgcolor='#FFFFFF')
                processed_image.save(output_path)

            if draw_guidelines:
                test_image, detected_faces = detect_faces(output_path)
                height, width, _ = test_image.shape
                print(f'shape0 {height} shape1 {width}')

                if len(detected_faces) == 0:
                    print(f'Face not detected in {filename}')
                else:
                    draw_guidelines_on_faces(test_image, detected_faces, output_path)

if __name__ == "__main__":
    input_directory = "./test_images"
    output_directory = "./output_images_directory"

    process_directory(input_directory, output_directory, draw_guidelines=True)