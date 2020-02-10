import numpy as np
import cv2

cam = cv2.VideoCapture(0)
while True:
    ret, frame = cam.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([110, 65, 75])
    high = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower, high)
    img = cv2.bitwise_and(frame, frame, mask=mask)
    kernel = np.ones((6, 6), np.uint8)
    erosion = cv2.erode(img, kernel, iterations=1)
    apertura = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)
    cierre = cv2.morphologyEx(apertura, cv2.MORPH_CLOSE, kernel)
    # gradient = cv2.morphologyEx(cierre, cv2.MORPH_GRADIENT, kernel)
    # Conv_hsv_Gray = cv2.cvtColor(cierre, cv2.COLOR_BGR2GRAY)
    # ret, mask2 = cv2.threshold(Conv_hsv_Gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    cierre[mask == 255] = [0, 100, 55]
    rgb = cv2.cvtColor(cierre, cv2.COLOR_HSV2BGR)
    cv2.imshow('Frames', frame)
    cv2.imshow('Objetos azules', img)
    cv2.imshow('Mascara', mask)
    cv2.imshow('gray', cierre)
    k = cv2.waitKey(1) & 0xFF
    if k == 2:
        break

cv2.destroyAllWindows()
cam.release()
