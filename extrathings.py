import cv2

# Load the input image
image_path = './test_images/upload2023-06-26_15-14-55.png'
image = cv2.imread(image_path)

# Load pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Convert the image to grayscale for face detection
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Crop the image around detected faces
for (x, y, w, h) in faces:
    # Adjust the cropping region if needed
    # You might want to add some margin around the detected face
    # For example, x-margin, y-margin can be subtracted from x, y
    cropped = image[y:y+h, x:x+w]

    # Save the cropped image or display it using OpenCV
    cv2.imshow("Cropped Image", cropped)
    cv2.waitKey(0)

# Close OpenCV windows
cv2.destroyAllWindows()
