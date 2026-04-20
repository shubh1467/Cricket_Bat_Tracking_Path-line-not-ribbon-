import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from collections import deque
from scipy.spatial.distance import pdist

# -----------------------------
# INIT
# -----------------------------
MODEL_PATH = "/home/ds-khel/Training_Bat_Model/runs/segment/runs_bat_seg/yolo26m_bat_seg_optimized/weights/best.pt"

VIDEO_PATH = "/home/ds-khel/Training_Bat_Model/vid.mp4"

OUTPUT = "/home/ds-khel/Training_Bat_Model/right_swing_analysis.mp4"

model = YOLO(MODEL_PATH)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

cap = cv2.VideoCapture(VIDEO_PATH)

fps = int(cap.get(cv2.CAP_PROP_FPS))
if fps <= 0:
    fps = 30

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(
    OUTPUT,
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (w,h)
)

# -----------------------------
# CALIBRATION
# -----------------------------
REAL_BAT_LENGTH = 0.85
meters_per_pixel = None

# -----------------------------
# TRACKING
# -----------------------------
trail = deque(maxlen=1000)

prev_point = None

speed_hist = deque(maxlen=10)

current_speed = 0.0
max_speed = 0.0
min_speed = float('inf')

prev_wrist = None
prev_direction = None
prev_toe = None

alpha_wrist = 0.85
alpha_dir   = 0.8
alpha_point = 0.7

# -----------------------------
# LOOP
# -----------------------------
while True:

    ret, frame = cap.read()

    if not ret:
        break

    vis = frame.copy()

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    result_pose = pose.process(rgb)

    if result_pose.pose_landmarks:

        landmarks = (
            result_pose
            .pose_landmarks
            .landmark
        )

        wrist = landmarks[
            mp_pose.PoseLandmark.RIGHT_WRIST
        ]

        wx,wy = (
            int(wrist.x*w),
            int(wrist.y*h)
        )

        current_wrist=np.array(
            [wx,wy]
        )

        # -------------------------
        # WRIST SMOOTHING
        # -------------------------

        if prev_wrist is not None:

            current_wrist=(
                alpha_wrist*prev_wrist
                +
                (1-alpha_wrist)*current_wrist
            )

        prev_wrist=current_wrist

        wx,wy=current_wrist.astype(int)

        # -------------------------
        # YOLO SEGMENTATION
        # -------------------------

        result_yolo=model(frame)[0]

        if (
            result_yolo.masks is None
            or
            len(result_yolo.masks.xy)==0
        ):
            prev_point=None
            out.write(vis)
            continue

        poly=(
            result_yolo
            .masks.xy[0]
            .astype(np.int32)
        )

        if cv2.contourArea(poly)<500:
            prev_point=None
            out.write(vis)
            continue

        mask=np.zeros(
            (h,w),
            dtype=np.uint8
        )

        cv2.fillPoly(
            mask,
            [poly],
            255
        )

        ys,xs=np.where(mask>0)

        points=np.stack(
            (xs,ys),
            axis=1
        )

        if len(points)>20:

            pixel_length=np.max(
                pdist(points)
            )

            if 50<pixel_length<600:

                meters_per_pixel=(
                    REAL_BAT_LENGTH/
                    pixel_length
                )

        # -------------------------
        # CENTROID
        # -------------------------

        M=cv2.moments(poly)

        if M["m00"]==0:
            continue

        cx=int(M["m10"]/M["m00"])
        cy=int(M["m01"]/M["m00"])

        centroid=np.array(
            [cx,cy]
        )

        # -------------------------
        # BAT DIRECTION
        # -------------------------

        direction=(
            centroid-current_wrist
        )

        norm=np.linalg.norm(direction)

        if norm>0:
            direction=direction/norm

        if prev_direction is not None:

            direction=(
                alpha_dir*prev_direction
                +
                (1-alpha_dir)*direction
            )

            direction=(
                direction/
                (np.linalg.norm(direction)+1e-6)
            )

        prev_direction=direction

        # -------------------------
        # TOE
        # -------------------------

        vecs=points-current_wrist

        projections=(
            vecs@direction
        )

        k=max(
            10,
            int(0.05*len(points))
        )

        idxs=np.argsort(
            projections
        )[-k:]

        toe_cluster=points[idxs]

        toe=np.mean(
            toe_cluster,
            axis=0
        )

        if prev_toe is not None:

            toe=(
                alpha_point*prev_toe
                +
                (1-alpha_point)*toe
            )

        prev_toe=toe

        toe=tuple(
            toe.astype(int)
        )

        # -------------------------
        # SWING PATH
        # -------------------------

        trail.append(toe)

        for i in range(
            1,
            len(trail)
        ):

            cv2.line(
                vis,
                trail[i-1],
                trail[i],
                (0,255,255),
                5
            )

        # -------------------------
        # POINTS
        # -------------------------

        cv2.circle(
            vis,
            (wx,wy),
            6,
            (0,255,0),
            -1
        )

        cv2.circle(
            vis,
            toe,
            6,
            (0,0,255),
            -1
        )

        # -------------------------
        # SPEED
        # -------------------------

        if (
            prev_point is not None
            and
            meters_per_pixel is not None
        ):

            pixel_dist=np.linalg.norm(
                np.array(toe)
                -
                np.array(prev_point)
            )

            speed_mps=(
                pixel_dist*
                fps*
                meters_per_pixel
            )

            speed_kmph=(
                speed_mps*3.6
            )

            speed_hist.append(
                speed_kmph
            )

            current_speed=np.mean(
                speed_hist
            )

            max_speed=max(
                max_speed,
                current_speed
            )

            min_speed=min(
                min_speed,
                current_speed
            )

        prev_point=toe

    # -------------------------
    # ALWAYS DRAW SPEEDS
    # (draw every frame)
    # -------------------------

    cv2.putText(
        vis,
        f"Speed: {current_speed:.1f} km/h",
        (20,50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,255),
        2
    )

    cv2.putText(
        vis,
        f"Max: {max_speed:.1f} km/h",
        (20,90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,0),
        2
    )

    if min_speed != float('inf'):

        cv2.putText(
            vis,
            f"Min: {min_speed:.1f} km/h",
            (20,130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,0,255),
            2
        )

    out.write(vis)

# -----------------------------
# CLEANUP
# -----------------------------

cap.release()
out.release()

# -----------------------------
# FINAL TERMINAL OUTPUT
# -----------------------------

if min_speed == float('inf'):
    min_speed = 0.0

print("\n=================================")
print("BAT SWING SPEED SUMMARY")
print("=================================")

print(f"Max Bat Speed : {max_speed:.2f} km/h")
print(f"Min Bat Speed : {min_speed:.2f} km/h")

print("Video saved to:")
print(OUTPUT)

print("DONE")