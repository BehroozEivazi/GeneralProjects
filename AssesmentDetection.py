import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('assesments.jpeg')


def display(img):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    new_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(new_img)
    plt.show()


display(img)

plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_russian_plate_number.xml')

def detect_plate(img):
    plate_image = img.copy()
    plate_rects = plate_cascade.detectMultiScale(plate_image, scaleFactor=1.3, minNeighbors=3)

    for (x, y, w, h) in plate_rects:
        cv2.rectangle(plate_image, (x, y), (x + w, y + h), (0, 0, 255), 4)

    return plate_image


result = detect_plate(img)
display(result)


def detect_and_blur_plate(img):
    plate_img = img.copy()
    roi = img.copy()
    plate_rects = plate_cascade.detectMultiScale(plate_img, scaleFactor=1.3, minNeighbors=3)
    for (x, y, w, h) in plate_rects:
        roi = roi[y:y + h, x:x + w]
        blurred = cv2.medianBlur(roi, 7)
        plate_img[y:y + h, x:x + w] = (255, 0, 0)  # or you can replace with blurred

    return plate_img


result = detect_and_blur_plate(img)
display(result)
