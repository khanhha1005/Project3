import cv2
import numpy as np

def filter_lines_by_angle(hough_lines, angle_threshold):
    if hough_lines is None:
        return []
    
    filtered_lines = []
    for line in hough_lines:
        for x1, y1, x2, y2 in line:
            # Calculate the angle of the line
            theta = np.arctan2(y2 - y1, x2 - x1) * (180.0 / np.pi)
            # Only add lines that are within the specified angle threshold
            if abs(theta) > angle_threshold:
                filtered_lines.append(line)
    return filtered_lines

def select_white(image):
    # Convert to HSV for better color filtering
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Define range for white color
    lower_white = np.array([0, 0, 168], dtype=np.uint8)
    upper_white = np.array([172, 111, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_white, upper_white)
    return cv2.bitwise_and(image, image, mask=mask)

def process_image(image):
    white_image = select_white(image)
    gray = cv2.cvtColor(white_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def region_of_interest(image):
    height, width = image.shape[:2]
    polygons = np.array([
        [(0, height), (width, height), (width, height//2), (0, height//2)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def average_slope_intercept(lines, image):
    left_lines = []  # Lines on the left of the lane
    right_lines = []  # Lines on the right of the lane

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            if 0.3 < abs(slope) < 2:  # Filter out lines with an unreasonable slope
                if slope < 0:
                    left_lines.append((slope, intercept))
                else:
                    right_lines.append((slope, intercept))

    left_avg = np.average(left_lines, axis=0) if left_lines else None
    right_avg = np.average(right_lines, axis=0) if right_lines else None
    left_line = make_line(image, left_avg) if left_avg is not None else None
    right_line = make_line(image, right_avg) if right_avg is not None else None
    return left_line, right_line

def make_line(image, line_parameters):
    if line_parameters is None:
        return None
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * 0.45)
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def draw_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines[0] is not None:
        cv2.line(line_image, (lines[0][0], lines[0][1]), (lines[0][2], lines[0][3]), (255, 0, 0), 10)
    if lines[1] is not None:
        cv2.line(line_image, (lines[1][0], lines[1][1]), (lines[1][2], lines[1][3]), (255, 0, 0), 10)
    return cv2.addWeighted(image, 0.8, line_image, 1, 1)


def process_frame(frame):
    canny_image = process_image(frame)
    roi_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(roi_image, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    angle_threshold = 10  # adjust as necessary
    filtered_lines = filter_lines_by_angle(lines, angle_threshold)
    averaged_lines = average_slope_intercept(filtered_lines, frame)
    return averaged_lines
