import numpy as np
import mediapipe as mp
import cv2
import time

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

# --- TRACKING INITIALIZATION ---
session_start = time.time()
look_away_frames = 0
total_frames = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1) # Mirror for natural feel
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    
    total_frames += 1
    gaze_status = "Center"
    is_off_screen = False

    if results.multi_face_landmarks:
        mesh_points = np.array([np.multiply([p.x, p.y], [w, h]).astype(int) 
                               for p in results.multi_face_landmarks[0].landmark])

        # Iris Landmarks: Left Eye (468), Right Eye (473)
        # We'll use the Right Eye (from camera perspective) for simplicity
        (re_x, re_y), re_radius = cv2.minEnclosingCircle(mesh_points[468:473])
        center_eye = np.array([re_x, re_y], dtype=np.int32)
        
        # Eye corners for reference
        # 33 = Left corner, 133 = Right corner, 159 = Top, 145 = Bottom
        lc = mesh_points[133]
        rc = mesh_points[33]
        tc = mesh_points[159]
        bc = mesh_points[145]

        # Calculate Ratios
        # Horizontal: distance to right corner / total width
        h_dist = np.linalg.norm(center_eye - rc)
        h_total = np.linalg.norm(lc - rc)
        h_ratio = h_dist / h_total

        # Vertical: distance to top / total height
        v_dist = np.linalg.norm(center_eye - tc)
        v_total = np.linalg.norm(bc - tc)
        v_ratio = v_dist / v_total

        # Thresholds (Adjust these based on your monitor distance)
        if h_ratio < 0.40:
            gaze_status = "Looking Right"
            is_off_screen = True
        elif h_ratio > 0.75:
            gaze_status = "Looking Left"
            is_off_screen = True
        elif v_ratio < 0.35:
            gaze_status = "Looking Up"
            is_off_screen = True
        elif v_ratio > 1.1: # Looking down (towards lap/phone)
            gaze_status = "Looking Down"
            is_off_screen = True
            
        # Visual Aid: Draw iris center
        cv2.circle(frame, center_eye, 2, (0, 255, 0), -1)
    else:
        gaze_status = "Face Not Detected"
        is_off_screen = True

    if is_off_screen:
        look_away_frames += 1

    # --- CALCULATION & DISPLAY ---
    # Probability logic: ratio of "suspicious" frames to total frames
    cheating_prob = (look_away_frames / total_frames) * 100
    
    # Colors: Green for safe, Red for high probability
    color = (0, 255, 0) if cheating_prob < 30 else (0, 0, 255)

    cv2.putText(frame, f"Gaze: {gaze_status}", (30, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(frame, f"Cheating Prob: {cheating_prob:.1f}%", (30, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Proctor AI", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()