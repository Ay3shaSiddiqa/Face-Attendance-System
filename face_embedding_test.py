import cv2
import numpy as np

print("Loading AI Models...")
# 1. Load the Face Detection AI (YuNet)
# The (320, 320) is just a default starting size; we will update it based on your webcam's resolution later.
detector = cv2.FaceDetectorYN.create("face_detection_yunet_2023mar.onnx", "", (320, 320))

# 2. Load the Face Recognition AI (SFace)
recognizer = cv2.FaceRecognizerSF.create("face_recognition_sface_2021dec.onnx", "")

print("Opening webcam... Please look at the camera.")
camera = cv2.VideoCapture(0)

while True:
    success, frame = camera.read()
    if not success:
        print("Could not read webcam.")
        break

    # Get the width and height of the webcam frame
    height, width, _ = frame.shape
    # Tell the detector the exact size of our image
    detector.setInputSize((width, height))

    # 3. Detect faces in the frame
    _, faces = detector.detect(frame)

    # If at least one face is found
    if faces is not None and len(faces) > 0:
        print("\nFace detected! Generating embedding...")
        
        # Grab the very first face found
        first_face = faces[0]
        
        # 4. Align the face (SFace requires the face to be straightened out first)
        aligned_face = recognizer.alignCrop(frame, first_face)
        
        # 5. Extract the embedding (the numerical representation of your face)
        embedding = recognizer.feature(aligned_face)
        
        print("SUCCESS! Your face was converted into a list of numbers.")
        print(f"The embedding has {embedding.shape[1]} numbers in it.")
        
        # Stop after successfully generating one embedding
        break

    cv2.imshow("Embedding Test", frame)
    
    # We also give you the option to quit manually just in case
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()