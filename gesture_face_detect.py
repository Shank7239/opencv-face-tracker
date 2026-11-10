import cv2
import numpy as np
import math
import time
import random

bg_subtractor = cv2.createBackgroundSubtractorKNN(detectShadows=False)
trail_points = []
show_fps = True

def resize(frame, width=1280):
    h = int(frame.shape[0] * (width / frame.shape[1]))
    return cv2.resize(frame, (width, h))

def normalize(frame):
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

def get_hand_points(frame):
    hsv = cv2.cvtColor(normalize(frame), cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70])
    upper_skin = np.array([20, 255, 255])
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    fg_mask = bg_subtractor.apply(frame)
    fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)[1]
    combined = cv2.bitwise_and(fg_mask, mask)
    combined = cv2.medianBlur(combined, 5)
    combined = cv2.dilate(combined, None, iterations=2)

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    hand = max(contours, key=cv2.contourArea)
    if cv2.contourArea(hand) < 5000:
        return []

    hull = cv2.convexHull(hand, returnPoints=True)
    return [tuple(pt[0]) for pt in hull]

def draw_spiderweb(frame, points, elapsed):
    global trail_points

    h, w = frame.shape[:2]
    cx = cy = None
    color_shift = int((math.sin(elapsed) + 1) * 127)

    if len(points) < 3:
        # Show pulsing border if no hand
        border_color = (50 + color_shift, 50, 50)
        thickness = int(10 + 5 * abs(math.sin(elapsed * 2)))
        cv2.rectangle(frame, (0, 0), (w, h), border_color, thickness)
        return frame

    # Center of hand
    cx = int(np.mean([x for x, _ in points]))
    cy = int(np.mean([y for _, y in points]))
    points.sort(key=lambda p: math.atan2(p[1] - cy, p[0] - cx))

    # Trail draw
    trail_points.append((cx, cy))
    if len(trail_points) > 30:
        trail_points = trail_points[-30:]
    for i, pt in enumerate(trail_points):
        alpha = int(255 * (i / len(trail_points)))
        cv2.circle(frame, pt, 3, (alpha, alpha // 2, alpha), -1)

    # Spiderweb lines
    for x, y in points:
        glow = min(int(math.hypot(x - cx, y - cy) / 2), 255)
        cv2.circle(frame, (x, y), 4, (255, glow, 255), -1)
        cv2.line(frame, (cx, cy), (x, y), (255, glow, 255), 1)

    # Rings
    for scale in [0.4, 0.7, 1.0]:
        ring = []
        for x, y in points:
            rx = int(cx + (x - cx) * scale)
            ry = int(cy + (y - cy) * scale)
            ring.append((rx, ry))
            cv2.circle(frame, (rx, ry), 2, (255, 255, 255), -1)
        for i in range(len(ring)):
            cv2.line(frame, ring[i], ring[(i + 1) % len(ring)], (255, 255, 255), 1)

    # Ripple center
    pulse_radius = int(20 + 5 * math.sin(elapsed * 5))
    cv2.circle(frame, (cx, cy), pulse_radius, (100, 100, 255), 1)

    return frame

# === Main ===
cap = cv2.VideoCapture(0)
cv2.namedWindow("🕸️ Spider Hand Tracker", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("🕸️ Spider Hand Tracker", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = resize(cv2.flip(frame, 1), width=1280)
    start = time.time()
    points = get_hand_points(frame)
    now = time.time()

    elapsed = now - prev_time
    prev_time = now

    frame = draw_spiderweb(frame, points, now)

    if show_fps:
        fps = int(1 / max(1e-5, (now - start)))
        cv2.putText(frame, f"FPS: {fps}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("🕸️ Spider Hand Tracker", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('f'):
        show_fps = not show_fps
    elif key == ord('c'):
        trail_points.clear()

cap.release()
cv2.destroyAllWindows()
