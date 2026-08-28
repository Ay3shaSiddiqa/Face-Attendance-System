import customtkinter as ctk
import os
import pandas as pd
import cv2
import numpy as np
from PIL import Image, ImageTk
from datetime import datetime
from tkinter import filedialog, messagebox

# Set up CustomTkinter theme
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class FaceAttendanceApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window configuration
        self.title("Face Recognition Attendance System")
        self.geometry("1150x700")
        self.minsize(950, 600)
        
        # Configure grid weight for responsiveness
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # --- Sidebar Frame ---
        self.sidebar_frame = ctk.CTkFrame(self, width=220, corner_radius=0, fg_color=("#1a1d24", "#14171d"))
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(5, weight=1)

        # Logo / App Title in Sidebar
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="BIOMETRIC\nATTENDANCE", font=ctk.CTkFont(family="Segoe UI", size=18, weight="normal"), text_color="#ffffff")
        self.logo_label.grid(row=0, column=0, padx=20, pady=(30, 20))

        # Navigation Buttons
        self.btn_register = ctk.CTkButton(self.sidebar_frame, text="Register Person", fg_color="transparent", text_color=("#dbdbdb", "#acb2c0"), hover_color=("#2b3240", "#2b3240"), anchor="w", command=self.show_register)
        self.btn_register.grid(row=1, column=0, padx=15, pady=8, sticky="ew")

        self.btn_recognition = ctk.CTkButton(self.sidebar_frame, text="Live Recognition", fg_color="transparent", text_color=("#dbdbdb", "#acb2c0"), hover_color=("#2b3240", "#2b3240"), anchor="w", command=self.show_recognition)
        self.btn_recognition.grid(row=2, column=0, padx=15, pady=8, sticky="ew")

        self.btn_attendance = ctk.CTkButton(self.sidebar_frame, text="Attendance Records", fg_color="transparent", text_color=("#dbdbdb", "#acb2c0"), hover_color=("#2b3240", "#2b3240"), anchor="w", command=self.show_attendance)
        self.btn_attendance.grid(row=3, column=0, padx=15, pady=8, sticky="ew")

        self.btn_settings = ctk.CTkButton(self.sidebar_frame, text="Settings", fg_color="transparent", text_color=("#dbdbdb", "#acb2c0"), hover_color=("#2b3240", "#2b3240"), anchor="w", command=self.show_settings)
        self.btn_settings.grid(row=4, column=0, padx=15, pady=8, sticky="ew")

        # --- Main Content Area ---
        self.content_area = ctk.CTkFrame(self, fg_color=("#21252f", "#1b1e28"), corner_radius=0)
        self.content_area.grid(row=0, column=1, sticky="nsew")
        self.content_area.grid_rowconfigure(0, weight=1)
        self.content_area.grid_columnconfigure(0, weight=1)

        # Initialize views container dictionary
        self.frames = {}
        
        # Build remaining views
        self.setup_register_view()
        self.setup_recognition_view()
        self.setup_attendance_view()
        self.setup_settings_view()
        
        # Camera & state variables
        self.reg_cap = None
        self.reg_running = False
        self.current_frame = None
        self.detected_faces_cache = None

        self.rec_cap = None
        self.rec_running = False
        self.session_marked = set()

        # Show Register Person by default on startup
        self.show_register()

    def reset_nav_buttons(self):
        for btn in [self.btn_register, self.btn_recognition, self.btn_attendance, self.btn_settings]:
            btn.configure(fg_color="transparent", text_color=("#dbdbdb", "#acb2c0"))

    def hide_all_frames(self):
        for frame in self.frames.values():
            frame.pack_forget()

    def setup_register_view(self):
        reg_frame = ctk.CTkFrame(self.content_area, fg_color="transparent")
        self.frames["Register"] = reg_frame

        title = ctk.CTkLabel(reg_frame, text="Register & Manage Persons", font=ctk.CTkFont(family="Segoe UI", size=24, weight="normal"), text_color="#ffffff")
        title.pack(anchor="w", padx=30, pady=(20, 5))

        body_grid = ctk.CTkFrame(reg_frame, fg_color="transparent")
        body_grid.pack(fill="both", expand=True, padx=30, pady=(0, 20))
        body_grid.grid_columnconfigure(0, weight=1)
        body_grid.grid_columnconfigure(1, weight=1)
        body_grid.grid_rowconfigure(0, weight=1)

        left_panel = ctk.CTkFrame(body_grid, fg_color=("#2b303c", "#232733"), corner_radius=14, border_width=1, border_color="#323846")
        left_panel.grid(row=0, column=0, padx=(0, 10), pady=0, sticky="nsew")

        lbl_reg_header = ctk.CTkLabel(left_panel, text="New Person Registration", font=ctk.CTkFont(family="Segoe UI", size=14, weight="normal"), text_color="#ffffff")
        lbl_reg_header.pack(anchor="w", padx=25, pady=(25, 5))

        lbl_name = ctk.CTkLabel(left_panel, text="Full Name", font=ctk.CTkFont(family="Segoe UI", size=12, weight="normal"), text_color="#8c93a3")
        lbl_name.pack(anchor="w", padx=25, pady=(15, 5))

        self.name_entry = ctk.CTkEntry(left_panel, placeholder_text="e.g. Ayesha Siddiqa", height=40, font=ctk.CTkFont(family="Segoe UI", size=13), fg_color="#1a1d24", border_color="#3b4252")
        self.name_entry.pack(fill="x", padx=25, pady=(0, 15))

        self.btn_capture = ctk.CTkButton(left_panel, text="Capture & Register Face", font=ctk.CTkFont(family="Segoe UI", size=14, weight="normal"), fg_color="#36a269", hover_color="#2e8b57", height=42, command=self.save_registration_face)
        self.btn_capture.pack(fill="x", padx=25, pady=5)

        self.reg_status_label = ctk.CTkLabel(left_panel, text="Status: Waiting for face input...", font=ctk.CTkFont(family="Segoe UI", size=12), text_color="#8c93a3", wraplength=330, justify="left")
        self.reg_status_label.pack(anchor="w", padx=25, pady=(15, 20))

        lbl_list_header = ctk.CTkLabel(left_panel, text="Registered Faces Manager", font=ctk.CTkFont(family="Segoe UI", size=13, weight="normal"), text_color="#ffffff")
        lbl_list_header.pack(anchor="w", padx=25, pady=(10, 5))

        self.reg_users_scroll = ctk.CTkScrollableFrame(left_panel, fg_color="#1a1d24", corner_radius=8, height=140)
        self.reg_users_scroll.pack(fill="x", padx=25, pady=(0, 20))

        right_panel = ctk.CTkFrame(body_grid, fg_color=("#2b303c", "#232733"), corner_radius=14, border_width=1, border_color="#323846")
        right_panel.grid(row=0, column=1, padx=(10, 0), pady=0, sticky="nsew")

        cam_title = ctk.CTkLabel(right_panel, text="Live Camera Preview", font=ctk.CTkFont(family="Segoe UI", size=14, weight="normal"), text_color="#ffffff")
        cam_title.pack(anchor="w", padx=20, pady=(25, 10))

        self.cam_preview_lbl = ctk.CTkLabel(right_panel, text="Camera Feed Inactive", fg_color="#14171d", corner_radius=10, text_color="#697086")
        self.cam_preview_lbl.pack(fill="both", expand=True, padx=20, pady=(0, 25))

    def load_registered_users_list(self):
        for widget in self.reg_users_scroll.winfo_children():
            widget.destroy()

        EMBED_DIR = "data/embeddings"
        if not os.path.exists(EMBED_DIR):
            lbl = ctk.CTkLabel(self.reg_users_scroll, text="No registered users found.", font=ctk.CTkFont(family="Segoe UI", size=11), text_color="#697086")
            lbl.pack(pady=10)
            return

        files = [f for f in os.listdir(EMBED_DIR) if f.endswith(".npy")]
        if not files:
            lbl = ctk.CTkLabel(self.reg_users_scroll, text="No registered users found.", font=ctk.CTkFont(family="Segoe UI", size=11), text_color="#697086")
            lbl.pack(pady=10)
            return

        for file_name in files:
            name = file_name.replace(".npy", "")
            row_f = ctk.CTkFrame(self.reg_users_scroll, fg_color="transparent")
            row_f.pack(fill="x", pady=3)

            lbl_name = ctk.CTkLabel(row_f, text=name, font=ctk.CTkFont(family="Segoe UI", size=12), text_color="#ffffff")
            lbl_name.pack(side="left", padx=5)

            btn_del = ctk.CTkButton(row_f, text="Delete", width=65, height=26, font=ctk.CTkFont(family="Segoe UI", size=11, weight="normal"), fg_color="#d85b67", hover_color="#c53b49", command=lambda n=name: self.delete_registered_person(n))
            btn_del.pack(side="right", padx=5)

    def delete_registered_person(self, name):
        if messagebox.askyesno("Confirm Deletion", f"Are you sure you want to delete '{name}' from the system?"):
            try:
                face_path = os.path.join("data/faces", f"{name}.jpg")
                embed_path = os.path.join("data/embeddings", f"{name}.npy")

                if os.path.exists(face_path):
                    os.remove(face_path)
                if os.path.exists(embed_path):
                    os.remove(embed_path)

                messagebox.showinfo("Success", f"Successfully deleted '{name}'!")
                self.load_registered_users_list()
            except Exception as e:
                messagebox.showerror("Error", f"Could not delete files: {str(e)}")

    def setup_recognition_view(self):
        rec_frame = ctk.CTkFrame(self.content_area, fg_color="transparent")
        self.frames["Recognition"] = rec_frame

        title = ctk.CTkLabel(rec_frame, text="Live Attendance Recognition", font=ctk.CTkFont(family="Segoe UI", size=24, weight="normal"), text_color="#ffffff")
        title.pack(anchor="w", padx=30, pady=(25, 5))

        subtitle = ctk.CTkLabel(rec_frame, text="Real-time face matching and automatic attendance logging system.", font=ctk.CTkFont(family="Segoe UI", size=13), text_color="#8c93a3")
        subtitle.pack(anchor="w", padx=30, pady=(0, 15))

        body_grid = ctk.CTkFrame(rec_frame, fg_color="transparent")
        body_grid.pack(fill="both", expand=True, padx=30, pady=(0, 25))
        body_grid.grid_columnconfigure(0, weight=3)
        body_grid.grid_columnconfigure(1, weight=2)
        body_grid.grid_rowconfigure(0, weight=1)

        rec_cam_panel = ctk.CTkFrame(body_grid, fg_color=("#2b303c", "#232733"), corner_radius=14, border_width=1, border_color="#323846")
        rec_cam_panel.grid(row=0, column=0, padx=(0, 10), pady=0, sticky="nsew")

        r_title = ctk.CTkLabel(rec_cam_panel, text="Webcam Stream", font=ctk.CTkFont(family="Segoe UI", size=13, weight="normal"), text_color="#ffffff")
        r_title.pack(anchor="w", padx=20, pady=(20, 10))

        self.rec_preview_lbl = ctk.CTkLabel(rec_cam_panel, text="Recognition Feed Inactive", fg_color="#14171d", corner_radius=10, text_color="#697086")
        self.rec_preview_lbl.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        info_panel = ctk.CTkFrame(body_grid, fg_color=("#2b303c", "#232733"), corner_radius=14, border_width=1, border_color="#323846")
        info_panel.grid(row=0, column=1, padx=(10, 0), pady=0, sticky="nsew")

        i_title = ctk.CTkLabel(info_panel, text="Recognition Status", font=ctk.CTkFont(family="Segoe UI", size=13, weight="normal"), text_color="#ffffff")
        i_title.pack(anchor="w", padx=20, pady=(20, 15))

        self.rec_status_card = ctk.CTkFrame(info_panel, fg_color="#14171d", corner_radius=10, border_width=1, border_color="#3b4252")
        self.rec_status_card.pack(fill="x", padx=20, pady=5)

        self.rec_status_title_lbl = ctk.CTkLabel(self.rec_status_card, text="STATUS: IDLE", font=ctk.CTkFont(family="Segoe UI", size=12, weight="normal"), text_color="#f3a35c")
        self.rec_status_title_lbl.pack(anchor="w", padx=15, pady=(15, 5))

        self.rec_name_lbl = ctk.CTkLabel(self.rec_status_card, text="Person: None Detected", font=ctk.CTkFont(family="Segoe UI", size=16, weight="normal"), text_color="#ffffff")
        self.rec_name_lbl.pack(anchor="w", padx=15, pady=5)

        self.rec_time_lbl = ctk.CTkLabel(self.rec_status_card, text="Time: --:--:--", font=ctk.CTkFont(family="Segoe UI", size=13), text_color="#8c93a3")
        self.rec_time_lbl.pack(anchor="w", padx=15, pady=(5, 15))

        self.rec_msg_lbl = ctk.CTkLabel(info_panel, text="Stand in front of the camera to automatically mark attendance.", font=ctk.CTkFont(family="Segoe UI", size=12), text_color="#8c93a3", wraplength=320, justify="left")
        self.rec_msg_lbl.pack(anchor="w", padx=20, pady=(20, 20))

    def setup_attendance_view(self):
        att_frame = ctk.CTkFrame(self.content_area, fg_color="transparent")
        self.frames["Attendance"] = att_frame

        title = ctk.CTkLabel(att_frame, text="Attendance Records", font=ctk.CTkFont(family="Segoe UI", size=24, weight="normal"), text_color="#ffffff")
        title.pack(anchor="w", padx=30, pady=(25, 5))

        subtitle = ctk.CTkLabel(att_frame, text="View logged attendance records and export them to Excel.", font=ctk.CTkFont(family="Segoe UI", size=13), text_color="#8c93a3")
        subtitle.pack(anchor="w", padx=30, pady=(0, 15))

        toolbar = ctk.CTkFrame(att_frame, fg_color="transparent")
        toolbar.pack(fill="x", padx=30, pady=(0, 10))

        self.btn_refresh_att = ctk.CTkButton(toolbar, text="Refresh Records", width=140, fg_color="#3b82f6", hover_color="#2563eb", command=self.load_attendance_table)
        self.btn_refresh_att.pack(side="left", padx=(0, 10))

        self.btn_export_excel = ctk.CTkButton(toolbar, text="Download Excel (.xlsx)", width=160, fg_color="#36a269", hover_color="#2e8b57", command=self.export_to_excel)
        self.btn_export_excel.pack(side="left")

        self.table_card = ctk.CTkFrame(att_frame, fg_color=("#2b303c", "#232733"), corner_radius=14, border_width=1, border_color="#323846")
        self.table_card.pack(fill="both", expand=True, padx=30, pady=(0, 25))

        self.table_scroll = ctk.CTkScrollableFrame(self.table_card, fg_color="transparent")
        self.table_scroll.pack(fill="both", expand=True, padx=10, pady=10)

        self.load_attendance_table()

    def load_attendance_table(self):
        for widget in self.table_scroll.winfo_children():
            widget.destroy()

        csv_path = "attendance/attendance.csv"
        if not os.path.exists(csv_path):
            lbl_empty = ctk.CTkLabel(self.table_scroll, text="No attendance records found yet. Start recognition to log records.", font=ctk.CTkFont(family="Segoe UI", size=13), text_color="#8c93a3")
            lbl_empty.pack(pady=40)
            return

        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                lbl_empty = ctk.CTkLabel(self.table_scroll, text="Attendance log file is currently empty.", font=ctk.CTkFont(family="Segoe UI", size=13), text_color="#8c93a3")
                lbl_empty.pack(pady=40)
                return

            header_frame = ctk.CTkFrame(self.table_scroll, fg_color="#1a1d24", corner_radius=6, height=38)
            header_frame.pack(fill="x", pady=(0, 6))
            
            for col_idx in range(4):
                header_frame.grid_columnconfigure(col_idx, weight=1, uniform="col")

            cols = ["Name", "Date", "Time", "Status"]
            for idx, col_name in enumerate(cols):
                lbl = ctk.CTkLabel(header_frame, text=col_name, font=ctk.CTkFont(family="Segoe UI", size=12, weight="normal"), text_color="#8c93a3")
                lbl.grid(row=0, column=idx, padx=15, pady=8, sticky="w")

            for i, row in df.iterrows():
                row_bg = "#232733" if i % 2 == 0 else "#2b303c"
                row_frame = ctk.CTkFrame(self.table_scroll, fg_color=row_bg, corner_radius=6, height=38)
                row_frame.pack(fill="x", pady=2)
                
                for col_idx in range(4):
                    row_frame.grid_columnconfigure(col_idx, weight=1, uniform="col")

                ctk.CTkLabel(row_frame, text=str(row.get("Name", "")), font=ctk.CTkFont(family="Segoe UI", size=12), text_color="#ffffff").grid(row=0, column=0, padx=15, pady=8, sticky="w")
                ctk.CTkLabel(row_frame, text=str(row.get("Date", "")), font=ctk.CTkFont(family="Segoe UI", size=12), text_color="#acb2c0").grid(row=0, column=1, padx=15, pady=8, sticky="w")
                ctk.CTkLabel(row_frame, text=str(row.get("Time", "")), font=ctk.CTkFont(family="Segoe UI", size=12), text_color="#acb2c0").grid(row=0, column=2, padx=15, pady=8, sticky="w")
                
                status_val = str(row.get("Status", "Present"))
                status_color = "#36a269" if status_val == "Present" else "#f3a35c"
                ctk.CTkLabel(row_frame, text=status_val, font=ctk.CTkFont(family="Segoe UI", size=12, weight="normal"), text_color=status_color).grid(row=0, column=3, padx=15, pady=8, sticky="w")

        except Exception as e:
            lbl_err = ctk.CTkLabel(self.table_scroll, text=f"Error loading attendance table: {str(e)}", font=ctk.CTkFont(family="Segoe UI", size=12), text_color="#d85b67")
            lbl_err.pack(pady=20)

    def export_to_excel(self):
        csv_path = "attendance/attendance.csv"
        if not os.path.exists(csv_path):
            messagebox.showerror("Export Error", "No attendance data records available to export.")
            return

        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                messagebox.showwarning("Warning", "The attendance log is empty.")
                return

            file_path = filedialog.asksaveasfilename(
                defaultextension=".xlsx",
                filetypes=[("Excel Workbook", "*.xlsx"), ("All Files", "*.*")],
                title="Save Attendance As Excel"
            )

            if file_path:
                df.to_excel(file_path, index=False)
                messagebox.showinfo("Success", f"Excel file exported successfully to:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Export Failed", f"Could not export Excel file: {str(e)}")

    def setup_settings_view(self):
        set_frame = ctk.CTkFrame(self.content_area, fg_color="transparent")
        self.frames["Settings"] = set_frame

        title = ctk.CTkLabel(set_frame, text="Application Settings", font=ctk.CTkFont(family="Segoe UI", size=24, weight="normal"), text_color="#ffffff")
        title.pack(anchor="w", padx=30, pady=(25, 5))

        subtitle = ctk.CTkLabel(set_frame, text="Configure system options and environment parameters.", font=ctk.CTkFont(family="Segoe UI", size=13), text_color="#8c93a3")
        subtitle.pack(anchor="w", padx=30, pady=(0, 15))

        card = ctk.CTkFrame(set_frame, fg_color=("#2b303c", "#232733"), corner_radius=14, border_width=1, border_color="#323846")
        card.pack(fill="x", padx=30, pady=10)

        lbl_engine = ctk.CTkLabel(card, text="AI Face Recognition Engine", font=ctk.CTkFont(family="Segoe UI", size=14, weight="normal"), text_color="#ffffff")
        lbl_engine.pack(anchor="w", padx=25, pady=(20, 5))

        lbl_engine_desc = ctk.CTkLabel(card, text="Using OpenCV YuNet (Face Detection) + SFace (128-d Embedding & Cosine Similarity Matcher).\nCompatible with Python 3.14 on Windows (No external TensorFlow/dlib requirements).", font=ctk.CTkFont(family="Segoe UI", size=12), text_color="#8c93a3", justify="left")
        lbl_engine_desc.pack(anchor="w", padx=25, pady=(0, 20))

    def start_registration_camera(self):
        if self.reg_cap is None or not self.reg_cap.isOpened():
            self.reg_cap = cv2.VideoCapture(0)
            self.reg_running = True
            self.load_registered_users_list()
            self.update_registration_feed()

    def stop_registration_camera(self):
        self.reg_running = False
        if self.reg_cap is not None:
            self.reg_cap.release()
            self.reg_cap = None

    def update_registration_feed(self):
        if self.reg_running and self.reg_cap and self.reg_cap.isOpened():
            ret, frame = self.reg_cap.read()
            if ret:
                self.current_frame = frame.copy()
                height, width, _ = frame.shape
                detector = cv2.FaceDetectorYN_create("face_detection_yunet_2023mar.onnx", "", (320, 320))
                detector.setInputSize((width, height))
                _, faces = detector.detect(frame)
                self.detected_faces_cache = faces

                display_frame = frame.copy()
                if faces is not None and len(faces) > 0:
                    face = faces[0]
                    box = list(map(int, face[:4]))
                    cv2.rectangle(display_frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (54, 162, 105), 2)

                cv2_image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv2_image)
                w = max(self.cam_preview_lbl.winfo_width(), 360)
                h = max(self.cam_preview_lbl.winfo_height(), 240)
                img = img.resize((w, h), Image.Resampling.LANCZOS)
                
                ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(w, h))
                self.cam_preview_lbl.configure(image=ctk_img, text="")
                self.cam_preview_lbl.image = ctk_img

            if self.reg_running:
                self.after(30, self.update_registration_feed)

    def save_registration_face(self):
        name = self.name_entry.get().strip()
        if not name:
            self.reg_status_label.configure(text="Error: Please enter a valid name first.", text_color="#d85b67")
            return

        if self.current_frame is None or self.detected_faces_cache is None:
            self.reg_status_label.configure(text="Error: Camera feed not active or ready.", text_color="#d85b67")
            return

        if len(self.detected_faces_cache) != 1:
            self.reg_status_label.configure(text="Error: Please ensure EXACTLY ONE face is clearly visible in the camera frame.", text_color="#d85b67")
            return

        try:
            FACE_DIR = "data/faces"
            EMBED_DIR = "data/embeddings"
            os.makedirs(FACE_DIR, exist_ok=True)
            os.makedirs(EMBED_DIR, exist_ok=True)

            recognizer = cv2.FaceRecognizerSF_create("face_recognition_sface_2021dec.onnx", "")
            aligned_face = recognizer.alignCrop(self.current_frame, self.detected_faces_cache[0])
            embedding = recognizer.feature(aligned_face)

            cv2.imwrite(os.path.join(FACE_DIR, f"{name}.jpg"), aligned_face)
            np.save(os.path.join(EMBED_DIR, f"{name}.npy"), embedding)

            self.reg_status_label.configure(text=f"Success! Face registered & saved for {name}.", text_color="#36a269")
            self.name_entry.delete(0, 'end')
            self.load_registered_users_list()
        except Exception as e:
            self.reg_status_label.configure(text=f"Error saving data: {str(e)}", text_color="#d85b67")

    def start_recognition_camera(self):
        if self.rec_cap is None or not self.rec_cap.isOpened():
            self.rec_cap = cv2.VideoCapture(0)
            self.rec_running = True
            self.update_recognition_feed()

    def stop_recognition_camera(self):
        self.rec_running = False
        if self.rec_cap is not None:
            self.rec_cap.release()
            self.rec_cap = None

    def update_recognition_feed(self):
        if self.rec_running and self.rec_cap and self.rec_cap.isOpened():
            ret, frame = self.rec_cap.read()
            if ret:
                height, width, _ = frame.shape
                
                detector = cv2.FaceDetectorYN_create("face_detection_yunet_2023mar.onnx", "", (320, 320))
                recognizer = cv2.FaceRecognizerSF_create("face_recognition_sface_2021dec.onnx", "")
                detector.setInputSize((width, height))

                registered_faces = {}
                EMBED_DIR = "data/embeddings"
                if os.path.exists(EMBED_DIR):
                    for file_name in os.listdir(EMBED_DIR):
                        if file_name.endswith(".npy"):
                            name = file_name.replace(".npy", "")
                            embedding = np.load(os.path.join(EMBED_DIR, file_name))
                            registered_faces[name] = embedding

                _, faces = detector.detect(frame)
                display_frame = frame.copy()
                MATCH_THRESHOLD = 0.363

                detected_name = "Unknown"
                best_score = 0.0

                if faces is not None and len(faces) > 0:
                    for face in faces:
                        aligned_face = recognizer.alignCrop(frame, face)
                        live_embedding = recognizer.feature(aligned_face)

                        for name, saved_embedding in registered_faces.items():
                            score = recognizer.match(live_embedding, saved_embedding, cv2.FaceRecognizerSF_FR_COSINE)
                            if score > best_score:
                                best_score = score
                                if score >= MATCH_THRESHOLD:
                                    detected_name = name

                        box = list(map(int, face[:4]))
                        color = (54, 162, 105) if detected_name != "Unknown" else (93, 91, 216)
                        cv2.rectangle(display_frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), color, 2)
                        cv2.putText(display_frame, f"{detected_name} ({best_score:.2f})", (box[0], box[1] - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                    if detected_name != "Unknown":
                        self.rec_status_title_lbl.configure(text="STATUS: RECOGNIZED", text_color="#36a269")
                        self.rec_name_lbl.configure(text=f"Person: {detected_name}")
                        current_time_str = datetime.now().strftime("%H:%M:%S")
                        self.rec_time_lbl.configure(text=f"Time: {current_time_str}")
                        self.mark_in_app_attendance(detected_name)
                    else:
                        self.rec_status_title_lbl.configure(text="STATUS: UNKNOWN FACE", text_color="#f3a35c")
                        self.rec_name_lbl.configure(text="Person: Unrecognized")
                        self.rec_time_lbl.configure(text="Time: --:--:--")
                else:
                    self.rec_status_title_lbl.configure(text="STATUS: SCANNING", text_color="#3b82f6")
                    self.rec_name_lbl.configure(text="Person: No face detected")
                    self.rec_time_lbl.configure(text="Time: --:--:--")

                cv2_image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv2_image)
                w = max(self.rec_preview_lbl.winfo_width(), 450)
                h = max(self.rec_preview_lbl.winfo_height(), 320)
                img = img.resize((w, h), Image.Resampling.LANCZOS)
                
                ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(w, h))
                self.rec_preview_lbl.configure(image=ctk_img, text="")
                self.rec_preview_lbl.image = ctk_img

            if self.rec_running:
                self.after(30, self.update_recognition_feed)

    def mark_in_app_attendance(self, name):
        ATTENDANCE_DIR = "attendance"
        os.makedirs(ATTENDANCE_DIR, exist_ok=True)
        ATTENDANCE_FILE = os.path.join(ATTENDANCE_DIR, "attendance.csv")
        
        today = datetime.now().strftime("%Y-%m-%d")
        now_time = datetime.now().strftime("%H:%M:%S")
        
        if name in self.session_marked:
            self.rec_msg_lbl.configure(text=f"Attendance already marked for {name} during this session.", text_color="#36a269")
            return
            
        already_marked = False
        file_exists = os.path.exists(ATTENDANCE_FILE)
        
        if file_exists:
            with open(ATTENDANCE_FILE, "r") as f:
                for line in f:
                    if name in line and today in line:
                        already_marked = True
                        break
        
        if not already_marked:
            with open(ATTENDANCE_FILE, "a") as f:
                if not file_exists:
                    f.write("Name,Date,Time,Status\n")
                f.write(f"{name},{today},{now_time},Present\n")
            self.rec_msg_lbl.configure(text=f"Success! Attendance marked for {name} at {now_time}.", text_color="#36a269")
        else:
            self.rec_msg_lbl.configure(text=f"{name} is already marked present for today.", text_color="#36a269")
            
        self.session_marked.add(name)

    def show_register(self):
        self.stop_recognition_camera()
        self.reset_nav_buttons()
        self.btn_register.configure(fg_color="#2b3240", text_color="#ffffff")
        self.hide_all_frames()
        self.load_registered_users_list()
        self.raise_frame("Register")
        self.start_registration_camera()

    def show_recognition(self):
        self.stop_recognition_camera()
        self.reset_nav_buttons()
        self.btn_recognition.configure(fg_color="#2b3240", text_color="#ffffff")
        self.hide_all_frames()
        self.raise_frame("Recognition")
        self.start_recognition_camera()

    def show_attendance(self):
        self.stop_recognition_camera()
        self.stop_recognition_camera()
        self.reset_nav_buttons()
        self.btn_attendance.configure(fg_color="#2b3240", text_color="#ffffff")
        self.hide_all_frames()
        self.load_attendance_table()
        self.raise_frame("Attendance")

    def show_settings(self):
        self.stop_recognition_camera()
        self.stop_recognition_camera()
        self.reset_nav_buttons()
        self.btn_settings.configure(fg_color="#2b3240", text_color="#ffffff")
        self.hide_all_frames()
        self.raise_frame("Settings")

    def raise_frame(self, name):
        frame = self.frames[name]
        frame.pack(fill="both", expand=True)

if __name__ == "__main__":
    app = FaceAttendanceApp()
    app.mainloop()