# %% [markdown]
# in machine learning and computer vision the process of making sense of visual data is often called inference or prediction 

# %%
"""import torch

print(f"PyTorch version: {torch.__version__}")

# Check if a GPU (NVIDIA) is available
cuda_available = torch.cuda.is_available()
print(f"Is CUDA (GPU) available? {cuda_available}")

if cuda_available:
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
else:
    print("Running on: CPU (This will be slower for video processing)")

# The 'Stress Test': Create a random tensor and move it to the device
try:
    device = torch.device("cuda" if cuda_available else "cpu")
    x = torch.rand(5, 3).to(device)
    print("Health Check: PASSED (Tensors are moving correctly)")
except Exception as e:
    print(f"Health Check: FAILED. Error: {e}")
"""
# %% [markdown]
# i am going to use an object detector called yolo

# %% [markdown]
# source is for -> where the video is and can give a url for youtube video 
# or live video using source = 0 the camera
# show -> to show results while we are doing inference

# %%
from ultralytics import YOLO
#yolov26 is the latest version
model = YOLO("yolo26n.pt")  # Load an official Detect model the nano version of it

# Perform tracking with the model
results = model.track(source = "tennis_video.mp4", show=True)  # Tracking with default tracker




# %% [markdown]
# from these results we can see that there is 214 images here 

# %%
from ultralytics import YOLO
#yolov26 is the latest version
model = YOLO("yolo26n.pt")  # Load an official Detect model the nano version of it
#.track differ from .predict as it tracks the object across the video frames to know that this object is the same as the one in the previous frame so we are tracking the object not just detecting them
results = model.track(source = "tennis_video.mp4", show=True, tracker="bytetrack.yaml", save = True, persist= True)  # with ByteTrack


# %% [markdown]
# Coordinate Mapping:
# the bounding boxes (around person, other objects)  mainly consistent of four values there is multiple ways to represent it like 
# 1- by the x-center and the y-center (the x position and the y position of the object) with the width and height of the rectangle(bound box)
# 2- represent it by the far ends of the rectangle called (xyxy)(in the video he called it x minimum y minimum, x maximum, y maximum)

# %%
for r in results.boxes:
    print(r)
    #boxes = r.boxes.xyxy  # Player locations
    #ids = r.boxes.id      # Player IDs

# %%
import cv2
# before yolo
cap = cv2.VideoCapture("tennis_video.mp4")
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

print(f"Correct Dimensions: {frame_width} (W) x {frame_height} (H)")

# %%

cap_after_yolo = cv2.VideoCapture("tennis_video.avi")
frame_width_after_yolo = int(cap_after_yolo.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height_after_yolo = int(cap_after_yolo.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap_after_yolo.release()

print(f"Correct Dimensions: {frame_width_after_yolo} (W) x {frame_height_after_yolo} (H)")

# %% [markdown]
# ### YOLO26 return the coordinates relative to the original input size automatically

# %% [markdown]
# getting the IDS and locations of only the players and save them for the heatmap 

# %%
import numpy as np

# Store coordinates for the heatmap
player_positions = []
# Define the court area (Example coordinates - adjust to fit your 1920x1080 frame)these numbers worked perfectly fine
court_poly = np.array([[100, 1000], [1820, 1000], [1350, 200], [570, 200]], np.int32) 

for result in results:
    # Create a copy of the frame to draw on (if using results[i].orig_img)
    frame = result.orig_img.copy()

    # Draw the polygon so you can see it in the 'show' window
    cv2.polylines(frame, [court_poly], isClosed=True, color=(0, 255, 0), thickness=5)

    boxes = result.boxes
    for box in boxes:
        # 1. Get the class ID (0 is usually 'person')
        cls = int(box.cls[0])
            
        # 2. Only process if it's a person
        if cls == 0:
            # Get center (x, y) of the player bounding box
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            # Check if they are inside the court
            is_inside = cv2.pointPolygonTest(court_poly, (cx, cy), False)

            if is_inside >= 0:
               # This is a player, proceed with heatmap/tracking
               player_positions.append((cx, cy))
               
            # Optional: Draw a circle for points that were "Accepted"
               cv2.circle(frame, (cx, cy), 10, (0, 0, 255), -1)
        # Show the live progress
        cv2.imshow("Verifying Detection Zone", frame)
        # Press 'q' to stop the video early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()        

# %%

# Create blank image
heatmap_img = np.zeros((1080, 1920), dtype=np.float32)

# Populate heatmap
for pos in player_positions:
    cv2.circle(heatmap_img, pos, 25, 1, -1)

# Apply blur and color
heatmap_img = cv2.GaussianBlur(heatmap_img, (51, 51), 0)
heatmap_norm = cv2.normalize(heatmap_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)

# Save it!
cv2.imwrite("player_heatmap.png", heatmap_color)
print("Heatmap saved as player_heatmap2.png")

# %% [markdown]
# The heatmap shows where they were, but the analysis tells how they played.
# -Match Analysis is taking the raw (x, y) coordinates collected and turning them into stats.
# 
# Step 1: Calculate "Movement Coverage" (Distance)This is the most important "Quantitative Insight." You want to know how many meters the players ran.
# How to do it:Loop through the player_positions.
# Calculate the distance between the point in Frame 1 and the point in Frame 2 using the distance formula
# d = sqrt{(x_2-x_1)^2 + (y_2-y_1)^2}

# %% [markdown]
# # Match Analysis

# %%
# 1. Separate the data into two lists
top_player_path = []
bottom_player_path = []

# Assuming net is at y = 550
for pos in player_positions:
    if pos[1] < 550:
        top_player_path.append(pos)
    else:
        bottom_player_path.append(pos)

def calculate_path_dist(path):
    dist_px = 0
    for i in range(1, len(path)):
        p1 = path[i-1]
        p2 = path[i]
        d = np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
        # Filter only extreme tracking glitches (teleporting across the court)
        if d < 150: 
            dist_px += d
    return dist_px

# 2. Calculate for both
top_dist = calculate_path_dist(top_player_path)
bottom_dist = calculate_path_dist(bottom_player_path)

# 3. Final Results
total_meters = (top_dist + bottom_dist) / 90  # combined movement
baseline_presence = (len(bottom_player_path) / len(player_positions)) * 100
dominant_playstyle = 'Defensive/Baseline' if baseline_presence > 50 else 'Aggressive/Net'

print("======= FINAL MATCH ANALYSIS =======")
print(f"Total Distance Covered: {total_meters:.2f} meters")
print(f"style: {dominant_playstyle}")
print(f"Baseline Presence: {baseline_presence:.1f}%")

# %%
with open("analysis_report.txt", "w") as f:
    f.write(f"Match Analysis Results\n")
    f.write(f"Distance: {total_meters:.2f}m\n")
    f.write(f"Style: {dominant_playstyle}\n")
    f.write(f"Baseline Presence: {baseline_presence:.1f}%")

# %% [markdown]
# we need a detector model to detect the ball as yolo is not detecting it 
# fine tune a yolo model on the balls (tennis balls that are moving very fast )
# we don't need a tracker for the ball only detector 

# %% [markdown]
# PyInstaller (the tool that makes the .exe) cannot read a Notebook file. It only reads standard Python scripts.

# %%



