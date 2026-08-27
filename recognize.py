import cv2
import numpy as np
import os
from datetime import datetime  # <--- Python's built-in date & time module

EMBED_DIR = "data/embeddings"
ATTENDANCE_DIR = "attendance"

# Automatically create the attendance folder
os.makedirs(ATTENDANCE_DIR, exist_ok=True)
ATTENDANCE_FILE = os.path.join(ATTENDANCE_DIR, "attendance.csv")

# 1. Load the OpenCV AI Models
detector = cv2.FaceDetectorYN_create("face_detection_yunet_2023mar.onnx", "", (320, 320))
recognizer = cv2.FaceRecognizerSF_create("face_recognition_sface_2021dec.onnx", "")

# 2. Load all registered faces into memory
registered_faces = {}
if os.path.exists(EMBED_DIR):
    for file_name in os.listdir(EMBED_DIR):
        if file_name.endswith(".npy"):
            name = file_name.replace(".npy", "")
            embedding = np.load(os.path.join(EMBED_DIR, file_name))
            registered_faces[name] = embedding

print(f"Loaded {len(registered_faces)} registered users.")

# Session tracking to prevent redundant file writes
session_marked = set()

def mark_attendance(name):
    # Using the datetime module to fetch current system date and time strings
    today = datetime.now().strftime("%Y-%m-%d")
    now_time = datetime.now().strftime("%H:%M:%S")
    
    if name in session_marked:
        return
        
    already_marked = False
    file_exists = os.path.exists(ATTENDANCE_FILE)
    
    # Check if a record for this date already exists in the CSV
    if file_exists:
        with open(ATTENDANCE_FILE, "r") as f:
            for line in f:
                if name in line and today in line:
                    already_marked = True
                    break
    
    # Write record if not already marked today
    if not already_marked:
        with open(ATTENDANCE_FILE, "a") as f:
            if not file_exists:
                f.write("Name,Date,Time,Status\n")
            f.write(f"{name},{today},{now_time},Present\n")
        print(f"--> Success! Attendance marked for {name} on {today} at {now_time}.")
    
    session_marked.add(name)

# 3. Open Webcam
cap = cv2.VideoCapture(0)
print("Starting recognition with date tracking... Press 'q' to QUIT.")

MATCH_THRESHOLD = 0.363 

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape
    detector.setInputSize((width, height))
    
    _, faces = detector.detect(frame)
    display_frame = frame.copy()

    if faces is not None:
        for face in faces:
            aligned_face = recognizer.alignCrop(frame, face)
            live_embedding = recognizer.feature(aligned_face)

            best_match_name = "Unknown"
            best_score = 0.0

            for name, saved_embedding in registered_faces.items():
                score = recognizer.match(live_embedding, saved_embedding, cv2.FaceRecognizerSF_FR_COSINE)
                if score > best_score:
                    best_score = score
                    if score >= MATCH_THRESHOLD:
                        best_match_name = name
                        mark_attendance(best_match_name) 

            box = list(map(int, face[:4]))
            color = (0, 255, 0) if best_match_name != "Unknown" else (0, 0, 255)
            
            cv2.rectangle(display_frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), color, 2)
            cv2.putText(display_frame, f"{best_match_name} ({best_score:.2f})", (box[0], box[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Face Recognition", display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()