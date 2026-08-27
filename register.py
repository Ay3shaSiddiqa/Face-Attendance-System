import cv2
import numpy as np
import os

# 1. Define storage folders
FACE_DIR = "data/faces"
EMBED_DIR = "data/embeddings"

# Automatically create the folders if they don't exist yet!
os.makedirs(FACE_DIR, exist_ok=True)
os.makedirs(EMBED_DIR, exist_ok=True)

# 2. Load the OpenCV AI Models (YuNet and SFace)
detector = cv2.FaceDetectorYN_create("face_detection_yunet_2023mar.onnx", "", (320, 320))
recognizer = cv2.FaceRecognizerSF_create("face_recognition_sface_2021dec.onnx", "")

# 3. Ask for the user's name
name = input("Enter the name of the person to register: ").strip()
if not name:
    print("Name cannot be empty. Exiting.")
    exit()

# 4. Open the webcam
cap = cv2.VideoCapture(0)
print(f"Starting camera for {name}... Press 's' to SAVE your face, or 'q' to QUIT.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Update detector with current frame size
    height, width, _ = frame.shape
    detector.setInputSize((width, height))

    # Detect faces
    _, faces = detector.detect(frame)
    display_frame = frame.copy()

    # Draw a green box around the face if detected
    if faces is not None and len(faces) > 0:
        face = faces[0]
        box = list(map(int, face[:4]))
        cv2.rectangle(display_frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0, 255, 0), 2)

    cv2.imshow("Registration - Press 's' to Save", display_frame)
    key = cv2.waitKey(1) & 0xFF

    # 5. Save the face and embedding if 's' is pressed
    if key == ord('s'):
        if faces is not None and len(faces) == 1:
            # Extract the 128-number face embedding
            aligned_face = recognizer.alignCrop(frame, faces[0])
            embedding = recognizer.feature(aligned_face)

            # Save the cropped face image
            img_path = os.path.join(FACE_DIR, f"{name}.jpg")
            cv2.imwrite(img_path, aligned_face)

            # Save the numerical embedding as a numpy file (.npy)
            embed_path = os.path.join(EMBED_DIR, f"{name}.npy")
            np.save(embed_path, embedding)

            print(f"Success! Face and embedding saved for {name}.")
            break
        else:
            print("Error: Make sure exactly ONE face is clearly in the frame!")
            
    elif key == ord('q'):
        print("Registration cancelled.")
        break

# 6. Cleanup
cap.release()
cv2.destroyAllWindows()