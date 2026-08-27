import tkinter as tk
from tkinter import messagebox
import subprocess
import os

def run_register():
    print("Opening Registration...")
    # This runs your register.py script in the background
    subprocess.Popen(["python", "register.py"])

def run_recognize():
    print("Starting Attendance Camera...")
    # This runs your recognize.py script in the background
    subprocess.Popen(["python", "recognize.py"])

def view_attendance():
    csv_path = os.path.abspath(os.path.join("attendance", "attendance.csv"))
    if os.path.exists(csv_path):
        print("Opening Attendance CSV...")
        # os.startfile is a Windows-specific command to open a file with its default program (like Excel or Notepad)
        os.startfile(csv_path)
    else:
        messagebox.showinfo("Info", "No attendance records found yet! Run the attendance camera first.")

# --- UI SETUP ---
root = tk.Tk()
root.title("Face Recognition Attendance")
root.geometry("450x350")
root.configure(bg="#f7f8fc") # Light background from your guide's CSS

# Title Label
title_label = tk.Label(root, text="FACE RECOGNITION ATTENDANCE", font=("Segoe UI", 14, "bold"), bg="#f7f8fc", fg="#40368e")
title_label.pack(pady=30)

# 1. Register Button
btn_register = tk.Button(root, text="[ Register Person ]", font=("Segoe UI", 12, "bold"), 
                         bg="#eeebff", fg="#5146b8", width=25, pady=5, command=run_register)
btn_register.pack(pady=10)

# 2. Attendance Button
btn_attendance = tk.Button(root, text="[ Start Attendance ]", font=("Segoe UI", 12, "bold"), 
                           bg="#eefaf3", fg="#36a269", width=25, pady=5, command=run_recognize)
btn_attendance.pack(pady=10)

# 3. View Records Button
btn_view = tk.Button(root, text="[ View Attendance ]", font=("Segoe UI", 12, "bold"), 
                     bg="#fff5e9", fg="#f3a35c", width=25, pady=5, command=view_attendance)
btn_view.pack(pady=10)

# Start the application loop
root.mainloop()