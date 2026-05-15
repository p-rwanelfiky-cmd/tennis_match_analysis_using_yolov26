import tkinter as tk
from tkinter import filedialog, Toplevel, Label, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO
import sys
import os

# 1. path function for exe file
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# --- AUTOMATIC POLYGON & BUFFER ---
def detect_court_polygon(first_frame):
    h, w, _ = first_frame.shape
    gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    
    # 1. Isolate bright pixels (court lines are white)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    # Restrict search area to lower 80% to avoid scoreboards and rafters
    mask_roi = np.zeros_like(thresh)
    mask_roi[int(h*0.2):int(h*0.95), int(w*0.05):int(w*0.95)] = 255
    thresh = cv2.bitwise_and(thresh, mask_roi)

    # 2. Extract structural lines
    edges = cv2.Canny(thresh, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=150, minLineLength=120, maxLineGap=20)
    
    if lines is None:
        return np.array([[100, h-50], [w-100, h-50], [int(w*0.7), int(h*0.25)], [int(w*0.3), int(h*0.25)]], np.int32)

    all_points = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        all_points.append((x1, y1))
        all_points.append((x2, y2))
        
    hull = cv2.convexHull(np.array(all_points, dtype=np.int32))
    
    # 3. Create a binary mask of this polygon and expand it (The Buffer Zone)
    poly_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(poly_mask, [hull], 255)
    
    kernel_size = int(w * 0.04) 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    expanded_mask = cv2.dilate(poly_mask, kernel, iterations=1)
    
    contours, _ = cv2.findContours(expanded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        epsilon = 0.02 * cv2.arcLength(contours[0], True)
        approx_poly = cv2.approxPolyDP(contours[0], epsilon, True)
        return approx_poly.reshape(-1, 2)
        
    return hull.reshape(-1, 2)


# --- THE CORE ANALYSIS FUNCTION ---
def run_analysis(video_path, status_label):
    try:
        
        # Use resource_path so the EXE knows where the file is
        MODEL_PATH = resource_path('yolo26n.pt') 
        model = YOLO(MODEL_PATH) 
        player_positions = []
        
        status_label.config(text="Status: Analyzing court geometry...", fg="purple")
        root.update()

        cap = cv2.VideoCapture(video_path)
        success, first_frame = cap.read()
        cap.release()
        
        if not success:
            raise Exception("Could not open or read the video file.")
            
        court_poly = detect_court_polygon(first_frame)
        h_frame, w_frame, _ = first_frame.shape
        mid_court = int(h_frame * 0.5)

        status_label.config(text="Status: Processing Video...", fg="blue")
        root.update() 

        # Force imgsz=1080 to maintain resolution on small objects
        results = model.track(source=video_path, stream=True, tracker="bytetrack.yaml", persist=True, imgsz=1080)

        # 2. Processing Loop
        for result in results:
            frame = result.orig_img.copy()
            cv2.polylines(frame, [court_poly], isClosed=True, color=(0, 255, 0), thickness=3)

            if result.boxes:
                for box in result.boxes:
                    cls = int(box.cls[0])
                    if cls == 0: # Person
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        
                        # Calculate center and feet coordinates
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)
                        feet_y = int(y2)
                        
                        # --- FIXED HYBRID STRATEGY (FLIPPED) ---
                        # Far away player (top half) -> Track by FEET
                        if cy < mid_court:
                            track_point = (cx, feet_y)
                        # Near player (bottom half) -> Track by CENTER
                        else:
                            track_point = (cx, cy)
                        
                        # Test the tailored track point against your court polygon
                        if cv2.pointPolygonTest(court_poly, track_point, False) >= 0:
                            player_positions.append(track_point)
                            cv2.circle(frame, track_point, 10, (0, 0, 255), -1)
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

            cv2.imshow("AIE 501 - Live Analysis", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

        if not player_positions:
            raise Exception("No players detected inside the court boundaries.")

        # 3. Match Analysis Calculations
        top_player_path = [p for p in player_positions if p[1] < mid_court]
        bottom_player_path = [p for p in player_positions if p[1] >= mid_court]

        def calc_dist(path):
            d_px = 0
            for i in range(1, len(path)):
                d = np.sqrt((path[i][0]-path[i-1][0])**2 + (path[i][1]-path[i-1][1])**2)
                if d < 150: d_px += d
            return d_px

        top_dist = calc_dist(top_player_path)
        bottom_dist = calc_dist(bottom_player_path)
        total_meters = (top_dist + bottom_dist) / 90
        baseline_pct = (len(bottom_player_path) / len(player_positions)) * 100 if player_positions else 0
        style = 'Defensive/Baseline' if baseline_pct > 50 else 'Aggressive/Net'

        # 4. Generate Heatmap
        heatmap_img = np.zeros((h_frame, w_frame), dtype=np.float32)
        for pos in player_positions:
            cv2.circle(heatmap_img, pos, 25, 1, -1)
        heatmap_img = cv2.GaussianBlur(heatmap_img, (51, 51), 0)
        heatmap_norm = cv2.normalize(heatmap_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
        cv2.imwrite("player_heatmap.png", heatmap_color)

        # 5. Save Report
        with open("analysis_report.txt", "w") as f:
            f.write(f"======= FINAL MATCH ANALYSIS =======\n")
            f.write(f"Total Distance Worked: {total_meters:.2f}m\n")
            f.write(f"Far Player: {top_dist/90:.2f}m | Near Player: {bottom_dist/90:.2f}m\n")
            f.write(f"Play Style Classification: {style}\n")
            f.write(f"Near Baseline Presence: {baseline_pct:.1f}%\n")

        status_label.config(text="Status: Analysis Complete!", fg="green")
        messagebox.showinfo("Project Result", f"Analysis Done!\nDistance: {total_meters:.2f}m\nStyle: {style}")

    except Exception as e:
        messagebox.showerror("Error", f"Failed: {str(e)}")
        status_label.config(text="Status: Error", fg="red")


# --- Function to show the Heatmap ---
def show_heatmap():
    try:
        new_win = Toplevel(root)
        new_win.title("Player Movement Heatmap")
        img_path = "player_heatmap.png"
        img = Image.open(img_path)
        img.thumbnail((800, 600)) 
        img_tk = ImageTk.PhotoImage(img)
        lbl = Label(new_win, image=img_tk)
        lbl.image = img_tk 
        lbl.pack(padx=10, pady=10)
    except FileNotFoundError:
        messagebox.showerror("Error", "Heatmap not found! Run the analysis first.")


# --- Function to show the Report ---
def show_report():
    try:
        with open("analysis_report.txt", "r") as f:
            report_content = f.read()
        messagebox.showinfo("Match Analysis Report", report_content)
    except FileNotFoundError:
        messagebox.showerror("Error", "Report not found! Run the analysis first.")


# --- GUI SETUP ---
def select_file():
    file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")])
    if file_path:
        file_label.config(text=f"File: {file_path.split('/')[-1]}")
        start_btn.config(state=tk.NORMAL, command=lambda: run_analysis(file_path, status_label))

root = tk.Tk()
root.title("Zewail City - Tennis Analytics (AIE 501)")
root.geometry("500x350")

tk.Label(root, text="Tennis Players Tracking System", font=("Arial", 16, "bold")).pack(pady=10)
tk.Button(root, text="Select Match Video", command=select_file, width=20).pack(pady=10)
file_label = tk.Label(root, text="No file selected", fg="gray")
file_label.pack()

status_label = tk.Label(root, text="Status: Waiting", font=("Arial", 10, "italic"))
status_label.pack(pady=20)

start_btn = tk.Button(root, text="START ANALYSIS", state=tk.DISABLED, bg="green", fg="white", font=("Arial", 12, "bold"), width=20)
start_btn.pack(pady=10)

button_frame = tk.Frame(root)
button_frame.pack(pady=10)

view_heatmap_btn = tk.Button(button_frame, text="View Heatmap", command=show_heatmap, width=15)
view_heatmap_btn.pack(side=tk.LEFT, padx=5)

view_report_btn = tk.Button(button_frame, text="View Stats", command=show_report, width=15)
view_report_btn.pack(side=tk.LEFT, padx=5)

root.mainloop()
