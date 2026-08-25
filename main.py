import argparse
import sys
import time

import cv2
import numpy as np

DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480
COUNT_LINE_ORIENTATION = "vertical"
COUNT_LINE_RATIO = 0.35
COUNT_LINE_TOLERANCE = 5
MIN_AREA = 700


def resize_keep_aspect(frame, width=DISPLAY_WIDTH, height=DISPLAY_HEIGHT):
    h, w = frame.shape[:2]
    scale = min(width / w, height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    y_offset = (height - new_h) // 2
    x_offset = (width - new_w) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return canvas


def open_capture(video_path=None):
    if video_path:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            return cap, f"video: {video_path}", True
        print(f"Cannot open video file: {video_path}")
        return None, None, False

    for index in range(5):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            return cap, f"camera {index}", False
        cap.release()

    return None, None, False


def get_center(x, y, w, h):
    return (x + w // 2, y + h // 2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vehicle counting demo")
    parser.add_argument("--video", type=str, default=None, help="Path to a video file. If omitted, it tries webcam indexes 0-4.")
    args = parser.parse_args()

    cap, source_info, is_video_file = open_capture(args.video)
    if cap is None:
        print("No camera or video source found.")
        print("Usage: python main.py --video /path/to/video.mp4")
        sys.exit(1)

    print(f"Using source: {source_info}")
    capture_fps = cap.get(cv2.CAP_PROP_FPS)
    if not capture_fps or capture_fps <= 1:
        capture_fps = 30.0
    frame_interval_ms = max(1, int(1000 / capture_fps))
    playback_start = time.perf_counter()
    frame_index = 0
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
    counted_ids = []
    vehicle_count = 0

    cv2.namedWindow("Vehicle Counting Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Vehicle Counting Detection", DISPLAY_WIDTH, DISPLAY_HEIGHT)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = resize_keep_aspect(frame)
        line_x = int(frame.shape[1] * COUNT_LINE_RATIO)
        line_y = int(frame.shape[0] * COUNT_LINE_RATIO)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        mask = bg_subtractor.apply(blur)
        _, thresh = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated = cv2.dilate(thresh, kernel, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        current_centers = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < MIN_AREA:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            center = get_center(x, y, w, h)
            current_centers.append((center, x, y, w, h))

        for center, x, y, w, h in current_centers:
            cx, cy = center
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

            if COUNT_LINE_ORIENTATION == "vertical":
                crossed = line_x - COUNT_LINE_TOLERANCE <= cx <= line_x + COUNT_LINE_TOLERANCE
            else:
                crossed = line_y - COUNT_LINE_TOLERANCE <= cy <= line_y + COUNT_LINE_TOLERANCE

            if crossed and center not in counted_ids:
                counted_ids.append(center)
                vehicle_count += 1

        if COUNT_LINE_ORIENTATION == "vertical":
            cv2.line(frame, (line_x, 0), (line_x, frame.shape[0]), (0, 0, 255), 2)
            cv2.putText(frame, "Crossing line", (line_x + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 0, 255), 2)
            cv2.putText(frame, "Crossing line", (20, max(30, line_y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.putText(frame, f"Vehicle Count: {vehicle_count}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Line: {COUNT_LINE_ORIENTATION}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Vehicle Counting Detection", frame)

        if is_video_file:
            frame_index += 1
            target_elapsed = frame_index / capture_fps
            actual_elapsed = time.perf_counter() - playback_start
            sleep_seconds = target_elapsed - actual_elapsed
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Final vehicle count: {vehicle_count}")
