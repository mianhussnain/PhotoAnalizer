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
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(200, 200))

    shoulders_roi = test_image  

    for (x, y, w, h) in faces:
        lowerAreaPercentage = ((test_image.shape[0]-(y+h)) / test_image.shape[0]) * 100  
        uperAreaPercentage = (y / test_image.shape[0]) * 100
        print(f"Lower area percentage: {lowerAreaPercentage:.2f}%")
        print(f"Uper area percentage: {uperAreaPercentage:.2f}%")
        print(f"y of image: {y:.2f}")
        margin = 100
    
        # Calculate new coordinates for the ROI
        x_roi = max(0, x - margin)
        y_roi = max(0, y - margin)
        w_roi = min(test_image.shape[1], w + 2 * margin)
        h_roi = min(test_image.shape[0], h + 2 * margin)

        # Crop the image to include the shoulders
        shoulders_roi = test_image[y_roi:y_roi + h_roi, x_roi:x_roi + w_roi]

        gray = cv2.cvtColor(shoulders_roi, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier('./cv2_cascade_classifier/haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(200, 200))

    return shoulders_roi, faces



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
                test_image, detected_faces = detect_faces(output_path)
                # print(f'test_image {test_image} detected_faces {detected_faces}')
            if len(detected_faces) == 0:
                    print(f'Face not detected in {filename}')
            else:
                white_background = Image.new("RGB", processed_image.size, "white")
                white_background.paste(Image.fromarray(test_image), (0, 0))
                white_background.save(output_path)

                if draw_guidelines:
                    draw_guidelines_on_faces(test_image, detected_faces, output_path)


if __name__ == "__main__":
    input_directory = "E:/server_images_bg"
    output_directory = "./output_images_directory"

    process_directory(input_directory, output_directory, draw_guidelines=True)