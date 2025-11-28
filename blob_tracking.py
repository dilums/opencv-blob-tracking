import cv2
import numpy as np
import sys


def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Could not open input video:", input_path)
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_index += 1

        # Slight contrast boost
        frame_float = frame.astype(np.float32) / 255.0
        frame_float = np.clip(frame_float * 1.05, 0, 1)
        base = (frame_float * 255).astype(np.uint8)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # Otsu threshold
        _, thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Clean noise
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        overlay = np.zeros_like(base)

        centers = []
        boxes = []
        areas = []

        min_area = (width * height) * 0.001

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            cx = x + w // 2
            cy = y + h // 2

            centers.append((cx, cy))
            boxes.append((x, y, w, h))
            areas.append(area)

        # Draw connections
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                x1, y1 = centers[i]
                x2, y2 = centers[j]

                dist = np.hypot(x2 - x1, y2 - y1)
                max_dist = np.hypot(width, height)
                fade = 1.0 - (dist / max_dist)
                if fade <= 0:
                    continue

                color = (
                    int(255 * fade),
                    int(255 * fade),
                    int(255 * fade),
                )
                cv2.line(
                    overlay,
                    (x1, y1),
                    (x2, y2),
                    color,
                    1,
                    lineType=cv2.LINE_AA,
                )

        # Draw boxes + labels
        for idx, ((x, y, w, h), center, area) in enumerate(
            zip(boxes, centers, areas)
        ):
            cx, cy = center

            cv2.rectangle(
                overlay,
                (x, y),
                (x + w, y + h),
                (255, 255, 255),
                1,
                lineType=cv2.LINE_AA,
            )

            corner_size = max(4, min(w, h) // 5)
            cv2.rectangle(
                overlay,
                (x, y),
                (x + corner_size, y + corner_size),
                (255, 255, 255),
                1,
                lineType=cv2.LINE_AA,
            )

            norm_area = area / (width * height)
            label = f"{idx:02d}  {norm_area:.4f}"

            text_pos = (x + 4, y - 6 if y - 6 > 10 else y + h + 14)

            cv2.putText(
                overlay,
                label,
                text_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
                lineType=cv2.LINE_AA,
            )

        # Blend overlay with base frame
        frame_final = cv2.addWeighted(base, 0.8, overlay, 0.8, 0)

        out.write(frame_final)

    cap.release()
    out.release()
    print("Saved:", output_path)

process_video("in.mp4", "out.mp4")