import cv2
import numpy as np

path = '/Users/poojaraghuram/Downloads/road.mp4'
video = cv2.VideoCapture(path)


def region(img):
    height, width = img.shape
    polygon = np.array([[(-width + 100, height), (900, 575), (1200, 575), (width, height)]])
    mask = np.zeros_like(img)
    mask = cv2.fillPoly(mask, polygon, 255)
    mask = cv2.bitwise_and(img, mask)
    return mask


def average_lines(img, lines):
    left = []
    right = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        params = np.polyfit((x1, x2), (y1, y2), 1)
        slope = params[0]
        y = params[1]
        if slope < 0:
            left.append((slope, y))
        else:
            right.append((slope, y))
    r_avg = np.average(right, 0)
    l_avg = np.average(left, 0)
    l_line = make_points(img, l_avg)
    r_line = make_points(img, r_avg)
    return np.array([l_line, r_line])


def make_points(img, avg):
    slope, y = avg
    y1 = img.shape[0]
    y2 = int(y1 * (3/5))
    x1 = int((y1 - y) // slope)
    x2 = int((y2 - y) // slope)
    return np.array([x1, y1, x2, y2])


def display_lines(img, lines):
    lines_img = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(lines_img, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return lines_img


while video.isOpened():
    ret, frame = video.read()

    grayscale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grayscale_img, (5, 5), 0)
    edges = cv2.Canny(blur, 100, 200)

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    l_yellow = np.array([20, 100, 100], "uint8")
    u_yellow = np.array([30, 255, 255], "uint8")

    yellow_mask = cv2.inRange(hsv_frame, l_yellow, u_yellow)
    white_mask = cv2.inRange(grayscale_img, 200, 255)
    combined_mask = cv2.bitwise_or(white_mask, yellow_mask)

    isolated = region(combined_mask)
    lines = cv2.HoughLinesP(isolated, 2, np.pi/180, 100, np.array([]), 40, 5)
    avg_lines = average_lines(frame, lines)
    black_lines = display_lines(frame, avg_lines)
    lanes = cv2.addWeighted(frame, 0.5, black_lines, 1, 1)

    # make suitable for quad view
    edges = np.stack((edges,) * 3, -1)
    isolated = np.stack((isolated,) * 3, -1)

    # Build quad view
    frame_top = np.hstack((frame, edges))
    frame_bot = np.hstack((isolated, lanes))
    frame_composite = np.vstack((frame_top, frame_bot))

    cv2.imshow('composite', frame_composite)

    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
