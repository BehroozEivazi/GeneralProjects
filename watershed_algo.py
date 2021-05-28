import cv2
import numpy as np
import matplotlib.pyplot as plt


def display(img, cmap='gray'):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')
    plt.show()


"""

sep_coins = cv2.imread('imgs/pennies.jpg')
display(sep_coins)

step haye ravesh
1-median blur
2-gray scale
3-binary threshold
4-find contours
sep_blur = cv2.medianBlur(sep_coins, ksize=25)
display(sep_blur)
gray_sep = cv2.cvtColor(sep_blur, cv2.COLOR_BGR2GRAY)
ret, sep_thresh = cv2.threshold(gray_sep, 160, 255, cv2.THRESH_BINARY_INV)
display(sep_thresh)

contours, hierarchy = cv2.findContours(sep_thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(sep_coins, contours, -1, (255, 0, 0), 10)
display(sep_coins)

"""

"""

with complex image 

"""

img = cv2.imread('imgs/pennies.jpg')
img_blur = cv2.medianBlur(img, ksize=35)
gray_img = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
ret, img_thresh = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY_INV)
display(img_thresh)
# toye image vaghti binary mishe ye noise haii hast ke mirine toye kar va ma az Otsu's method estefade mikonim
ret, img_thresh = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# display(img_thresh)
# noise removeall(optional)

kernel = np.ones((5, 5), np.uint8)

opening = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel, iterations=2)
sure_bg = cv2.dilate(opening, kernel=kernel, iterations=1)

# chize dige ke vojod dare mikhym peen haro az ham joda konim ke inja mirim soragh distance trasnform
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

display(unknown)

ret, markers = cv2.connectedComponents(sure_fg)

markers = markers + 1

markers[unknown == 255] = 0

markers = cv2.watershed(img, markers)

display(markers)
contours, hierarchy = cv2.findContours(markers, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img, contours, -1, (255, 0, 0), 10)
display(img)