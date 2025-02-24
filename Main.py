import cv2
import numpy as np
import math
from ultralytics import YOLO

def distance(p1, p2):
    """計算兩點距離"""
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def detect_inner_circle_by_thresholding(roi, outer_circle):
    """
    使用閾值分割來偵測內圓 (基於內圓黑色，外圓亮色的特性)
    """
    cx_o, cy_o, r_o = outer_circle
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # 使用 Otsu's Thresholding 自動選擇最佳閾值
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 形態學開運算去除雜訊
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    # 找出輪廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None

    best_circle = None
    best_dist = float('inf')

    for cnt in contours:
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        x, y, radius = int(x), int(y), int(radius)

        # 過濾過小或過大的圓
        if radius < 5 or radius > r_o * 0.8:
            continue

        # 計算圓心偏移量
        d = distance((x, y), (cx_o, cy_o))
        if d < best_dist:
            best_dist = d
            best_circle = (x, y, radius)

    return best_circle

def main():
    model = YOLO("yolov8-steel-pipe model.pt")

    img = cv2.imread("IMG_3830.JPG")
    if img is None:
        print("無法讀取影像。")
        return

    results = model.predict(source=img, conf=0.5, iou=0.45, max_det=20, imgsz=640)

    pipe_id = 0  

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        for box in boxes:
            pipe_id += 1
            x1, y1, x2, y2 = map(int, box[:4])

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            r_outer = (x2 - x1) // 2

            cv2.circle(img, (cx, cy), r_outer, (255, 0, 0), 2)  # 外圓 (藍色)
            cv2.circle(img, (cx, cy), 3, (0, 255, 255), -1)  # 外圓中心 (黃色)

            # ROI 區域
            roi = img[y1:y2, x1:x2]
            outer_circle = (cx - x1, cy - y1, r_outer)

            # 嘗試用閾值分割偵測內圓
            inner_circle = detect_inner_circle_by_thresholding(roi, outer_circle)
            if inner_circle is None:
                icx, icy, ir = outer_circle[0], outer_circle[1], int(outer_circle[2] * 0.7)  # 預設半徑
                method_used = "Default"
            else:
                icx, icy, ir = inner_circle
                method_used = "Thresholding"

            abs_icx, abs_icy = x1 + icx, y1 + icy

            cv2.circle(img, (abs_icx, abs_icy), ir, (0, 0, 255), 2)  # 內圓 (紅色)
            cv2.circle(img, (abs_icx, abs_icy), 3, (0, 255, 255), -1)  # 內圓中心 (黃色)

            cv2.putText(img, f"Pipe{pipe_id} In:{2*ir}px", (abs_icx+5, abs_icy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    cv2.imshow("Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
