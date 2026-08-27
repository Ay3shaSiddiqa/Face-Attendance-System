import cv2

# 1. Load OpenCV's built-in pretrained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 2. Open the webcam
camera = cv2.VideoCapture(0)

print("Starting webcam... Press 'q' to quit.")

while True:
    success, frame = camera.read()
    if not success:
        print("Could not read webcam.")
        break

    # 3. Convert the frame to grayscale (AI detects faces better in black and white)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 4. Detect faces in the image
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    # 5. Draw a green rectangle around every face detected
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # 6. Show the video
    cv2.imshow("Face Detection Test", frame)

    # 7. Quit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Clean up
camera.release()
cv2.destroyAllWindows()