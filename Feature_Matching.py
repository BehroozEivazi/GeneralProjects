import cv2
import numpy as np
import matplotlib.pyplot as plt


def display(img, cmap='gray'):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')
    plt.show()


reeses = cv2.imread('imgs/reeses_puffs.png', 0)

display(reeses)

cereals = cv2.imread('imgs/many_cereals.jpg', 0)
display(cereals)

orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(reeses, None)
kp2, des2 = orb.detectAndCompute(cereals, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(des1, des2)

# x.distance ye maghadiri mide ke az print gereftan bala mishe khond
matches = sorted(matches, key=lambda x: x.distance)

reeses_matches = cv2.drawMatches(reeses, kp1, cereals, kp2, matches[:25], None, flags=2)

display(reeses_matches)
"""
bala didim ke zamani ke shabahat ha ziad bood khob natoonest fe kone pas miam az method 
haye dige estefade mikonim SIFT descriptor

"""

sift = cv2.SIFT_create()

kp1, des1 = sift.detectAndCompute(reeses, None)
kp2, des2 = sift.detectAndCompute(cereals, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)
# inja be tedade 2 az behtarin noghat ke feature haye khobi hastan ro entekhab mikon
# ke avali behtarin noghtast va dovomi noghte dovome ke khobe
good = []

# oon 0.75 harcheghadr kam koni deghat ro mitooni afzaeyesh bedi
for match1, match2 in matches:
    if match1.distance < 0.75 * match2.distance:
        good.append([match1])

sift_matches = cv2.drawMatchesKnn(reeses, kp1, cereals, kp2, good, None, flags=2)

display(sift_matches)

"""
ye algo dige Flann Base matcher ee ke estefade mishe
Fast library for approximate nearest neighbors
ke sari tar az brud force ee inja mikhaym az oon estefade konim
"""

sift = cv2.SIFT_create()

kp1, des1 = sift.detectAndCompute(reeses, None)
kp2, des2 = sift.detectAndCompute(cereals, None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)
matchesMask = [[0, 0] for i in range(len(matches))]

good = []

# oon 0.75 harcheghadr kam koni deghat ro mitooni afzaeyesh bedi
for i, (match1, match2) in enumerate(matches):
    if match1.distance < 0.75 * match2.distance:
        matchesMask[i] = [1, 0]
        # good.append([match1])

draw_params = dict(matchColor=(0, 255, 0), singlePointColor=(255, 0, 0), matchesMask=matchesMask, flags=0)
flann_matches = cv2.drawMatchesKnn(reeses, kp1, cereals, kp2, matches, None, **draw_params)
# sift_matches = cv2.drawMatchesKnn(reeses, kp1, cereals, kp2, good, None, flags=2)

display(flann_matches)
