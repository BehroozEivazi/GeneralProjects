import cv2
import numpy as np
import matplotlib.pyplot as plt

"""with harris detector corner"""
flat_chess = cv2.imread('imgs/flat_chessboard.png')
flat_chess = cv2.cvtColor(flat_chess, cv2.COLOR_BGR2RGB)
# plt.imshow(flat_chess)
# plt.show()

gray_flat_chess = cv2.cvtColor(flat_chess, cv2.COLOR_BGR2GRAY)
# plt.imshow(gray_flat_chess, cmap='gray')
# plt.show()

real_chess = cv2.imread('imgs/real_chessboard.jpg')
real_chess = cv2.cvtColor(real_chess, cv2.COLOR_BGR2RGB)
# plt.imshow(real_chess)
# plt.show()

gray_real_chess = cv2.cvtColor(real_chess, cv2.COLOR_BGR2GRAY)
# plt.imshow(gray_real_chess, cmap='gray')
# plt.show()
"""with corner harris detector"""
# gray = np.float32(gray_flat_chess)
# dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
# dst = cv2.dilate(dst, None)
#
# flat_chess[dst > 0.01 * dst.max()] = [255, 0, 0]
#
# plt.imshow(flat_chess)
# plt.show()
#
# gray = np.float32(gray_real_chess)
# dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
# dst = cv2.dilate(dst, None)
#
# real_chess[dst > 0.01 * dst.max()] = [255, 0, 0]
#
# plt.imshow(real_chess)
# plt.show()

"""with shi_tomasi"""

corners = cv2.goodFeaturesToTrack(gray_flat_chess, 64, 0.01, 10, )
corners = np.int0(corners)
for i in corners:
    x, y = i.ravel()
    cv2.circle(flat_chess, (x, y), 3, (255, 0, 0), -1)
plt.imshow(flat_chess)
plt.show()


corners = cv2.goodFeaturesToTrack(gray_real_chess, 100, 0.01, 10, )
corners = np.int0(corners)
for i in corners:
    x, y = i.ravel()
    cv2.circle(real_chess, (x, y), 3, (255, 0, 0), -1)
plt.imshow(real_chess)
plt.show()
