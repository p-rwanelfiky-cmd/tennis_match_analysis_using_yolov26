import tkinter as tk
from tkinter import filedialog,Toplevel, Label, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO

# --- THE CORE ANALYSIS FUNCTION ---
def run_analysis(video_path, status_label):
    try:
        # 1. Initialize
        model = YOLO("yolo26n.pt")
        player_positions = []
        court_poly = np.array([[100, 1000], [1820, 1000], [1350, 200], [570, 200]], np.int32) 
        
        status_label.config(text="Status: Processing Video...", fg="blue")
        root.update() # Refresh GUI

        results = model.track(source=video_path, stream=True, tracker="bytetrack.yaml", persist=True)

        # 2. Processing Loop
        for result in results:
            frame = result.orig_img.copy()
            cv2.polylines(frame, [court_poly], isClosed=True, color=(0, 255, 0), thickness=5)

            if result.boxes:
                for box in result.boxes:
                    cls = int(box.cls[0])
                    if cls == 0: # Person
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                        
                        if cv2.pointPolygonTest(court_poly, (cx, cy), False) >= 0:
                            player_positions.append((cx, cy))
                            cv2.circle(frame, (cx, cy), 10, (0, 0, 255), -1)

            cv2.imshow("AIE 501 - Live Analysis", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

        # 3. Match Analysis Calculations
        top_player_path = [p for p in player_positions if p[1] < 550]
        bottom_player_path = [p for p in player_positions if p[1] >= 550]

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
        heatmap_img = np.zeros((1080, 1920), dtype=np.float32)
        for pos in player_positions:
            cv2.circle(heatmap_img, pos, 25, 1, -1)
        heatmap_img = cv2.GaussianBlur(heatmap_img, (51, 51), 0)
        heatmap_norm = cv2.normalize(heatmap_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
        cv2.imwrite("player_heatmap.png", heatmap_color)

        # 5. Save Report
        with open("analysis_report.txt", "w") as f:
            f.write(f"======= FINAL MATCH ANALYSIS =======\n")
            f.write(f"Total Distance: {total_meters:.2f}m\n")
            f.write(f"Djokovic: {top_dist/90:.2f}m | Sonego: {bottom_dist/90:.2f}m\n")
            f.write(f"Style: {style}\n")
            f.write(f"Baseline Presence: {baseline_pct:.1f}%")

        status_label.config(text="Status: Analysis Complete!", fg="green")
        messagebox.showinfo("Project Result", f"Analysis Done!\nDistance: {total_meters:.2f}m\nStyle: {style}")

    except Exception as e:
        messagebox.showerror("Error", f"Failed: {str(e)}")
        status_label.config(text="Status: Error", fg="red")


# --- 1. Function to show the Heatmap ---
def show_heatmap():
    try:
        # Create a new pop-up window
        new_win = Toplevel(root)
        new_win.title("Player Movement Heatmap")
        
        # Load the image using PIL
        img_path = "player_heatmap.png"
        img = Image.open(img_path)
        
        # Resize if it's too big for the screen
        img.thumbnail((800, 600)) 
        
        img_tk = ImageTk.PhotoImage(img)
        
        # We must keep a reference to the image so it doesn't disappear
        lbl = Label(new_win, image=img_tk)
        lbl.image = img_tk 
        lbl.pack(padx=10, pady=10)
        
    except FileNotFoundError:
        messagebox.showerror("Error", "Heatmap not found! Run the analysis first.")


# --- 2. Function to show the Report ---
def show_report():
    try:
        with open("analysis_report.txt", "r") as f:
            report_content = f.read()
        messagebox.showinfo("Match Analysis Report", report_content)
    except FileNotFoundError:
        messagebox.showerror("Error", "Report not found! Run the analysis first.")


# --- GUI SETUP ---
def select_file():
    file_path = filedialog.askopenfilename()
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
# Create a frame to hold the "View" buttons side-by-side
button_frame = tk.Frame(root)
button_frame.pack(pady=10)

view_heatmap_btn = tk.Button(button_frame, text="View Heatmap", command=show_heatmap, width=15)
view_heatmap_btn.pack(side=tk.LEFT, padx=5)

view_report_btn = tk.Button(button_frame, text="View Stats", command=show_report, width=15)
view_report_btn.pack(side=tk.LEFT, padx=5)
root.mainloop()