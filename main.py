import math

import sys

import cv2

import cv2 as cv

import numpy as np


# This function tries to find the 5-target pattern that looks like this
#  0  1  2
#  3     4
# The input is a list of (x,y) locations of possible targets, where each location is
# a numpy array of length 2. The output is a list of 5 targets in the proper order.
# If 5 targets in the correct configuration is not found, it returns an empty list.
def order_targets(allTargets):
    orderedTargets = []
    nTargets = len(allTargets)
    if nTargets < 5:
        return orderedTargets

    # Find 3 targets that are in a line.
    dMin = 1e9  # distance from a point to the midpt between points 1,3
    d02 = 0  # distance between points 1,3
    for i in range(0, nTargets):
        for j in range(i + 1, nTargets):
            # Get the mid point between i,j.
            midPt = (allTargets[i] + allTargets[j]) / 2

            # Find another target that is closest to this midpoint.
            for k in range(0, nTargets):
                if k == i or k == j:
                    continue
                d = np.linalg.norm(allTargets[k] - midPt)  # distance from midpoint
                if d < dMin:
                    dMin = d  # This is the minimum found so far; save it
                    i0 = i
                    i1 = k
                    i2 = j
                    d02 = np.linalg.norm(allTargets[i0] - allTargets[i2])

    # If the best distance from the midpoint is < 30% of the distance between
    # the two other points, then we probably have a colinear set; otherwise not.
    if dMin / d02 > 0.3:
        return orderedTargets  # return an empty list

    # We have found 3 colinear targets:  p0 -- p1 -- p2.
    # Now find the one closest to p0; call it p3.
    i3 = findClosest(allTargets, i0, excluded=[i0, i1, i2])
    if i3 is None:
        return []  # return an empty list

    # Now find the one closest to p2; call it p4.
    i4 = findClosest(allTargets, i2, excluded=[i0, i1, i2, i3])
    if i4 is None:
        return []  # return an empty list

    # Now, check to see where p4 is with respect to p0,p1,p2.  If the
    # signed area of the triangle p0-p2-p3 is negative, then we have
    # the correct order; ie
    #   0   1   2
    #   3     4
    # Otherwise we need to switch the order; ie
    #   2  1  0
    #   4     3

    # Signed area is the determinant of the 2x2 matrix [ p3-p0, p2-p0 ].
    p30 = allTargets[i3] - allTargets[i0]
    p20 = allTargets[i2] - allTargets[i0]
    M = np.array([[p30[0], p20[0]], [p30[1], p20[1]]])
    det = np.linalg.det(M)

    # Put the targets into the output list.
    if det < 0:
        orderedTargets.append(allTargets[i0])
        orderedTargets.append(allTargets[i1])
        orderedTargets.append(allTargets[i2])
        orderedTargets.append(allTargets[i3])
        orderedTargets.append(allTargets[i4])
    else:
        orderedTargets.append(allTargets[i2])
        orderedTargets.append(allTargets[i1])
        orderedTargets.append(allTargets[i0])
        orderedTargets.append(allTargets[i4])
        orderedTargets.append(allTargets[i3])

    return orderedTargets


# In the list of points "allPoints", find the closest point to point i0, that is not
# one of the points in the excluded list.  If none found, return None.
def findClosest(allPoints, i0, excluded):
    dMin = 1e9
    iClosest = None
    for i in range(0, len(allPoints)):
        if i in excluded:
            continue
        d = np.linalg.norm(allPoints[i] - allPoints[i0])
        if d < dMin:
            dMin = d
            iClosest = i
    return iClosest


def contains(r1, r2):
    return r1[0][0] < r2[0][0] < r2[1][0] < r1[1][0] and r1[0][1] < r2[0][1] < r2[1][1] < r1[1][1]


##################################################################################################################

# MAIN
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def main():
    index = 0

    pointsCCC = np.empty([5, 2])

    pointsObj = np.array([[-3.7, -2.275, 0], [0, -2.275, 0], [3.7, -2.275, 0], [-3.7, 2.275, 0], [3.7, 2.275, 0]])

    img = cv2.imread('CCCtarget.jpg')

    gray = cv.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray_inv = cv2.bitwise_not(gray)

    thresh1, binary_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    thresh2, binary_img_invert = cv2.threshold(gray_inv, 0, 255,

                                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

    filtered_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)  # OPENING FOLLOWED BY CLOSING

    filtered_img_inv = cv2.morphologyEx(binary_img_invert, cv2.MORPH_OPEN, kernel)

    output1 = cv2.connectedComponentsWithStats(filtered_img)  # White Blobs

    output2 = cv2.connectedComponentsWithStats(filtered_img_inv)  # Black Blobs

    (numLabels1, labels1, stats1, centroids1) = output1

    (numLabels2, labels2, stats2, centroids2) = output2

    # loop over the number of unique connected component labels

    for i in range(0, numLabels1):

        for j in range(0, numLabels2):

            # extract the connected component statistics and centroid for

            # the current label

            # white blobs

            x = stats1[i, cv2.CC_STAT_LEFT]

            y = stats1[i, cv2.CC_STAT_TOP]

            w = stats1[i, cv2.CC_STAT_WIDTH]

            h = stats1[i, cv2.CC_STAT_HEIGHT]

            area = stats1[i, cv2.CC_STAT_AREA]

            (cX, cY) = centroids1[i]

            # black blobs

            x2 = stats2[j, cv2.CC_STAT_LEFT]

            y2 = stats2[j, cv2.CC_STAT_TOP]

            w2 = stats2[j, cv2.CC_STAT_WIDTH]

            h2 = stats2[j, cv2.CC_STAT_HEIGHT]

            area2 = stats2[j, cv2.CC_STAT_AREA]

            (cX2, cY2) = centroids2[j]

            distance = math.sqrt(((cX2 - cX) ** 2) + ((cY2 - cY) ** 2))

            rect1 = ((x, y), (x + w, y + h))

            rect2 = ((x2, y2), (x2 + w2, y2 + h2))

            if distance < 1.0 and area < 200 and contains(rect1, rect2):
                # Draw a rectangle around the matched region.

                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)  # Box around matched regions

                cv2.drawMarker(img, (int(cX), int(cY)), (255, 0, 0))

                pointsCCC[index][0] = cX
                pointsCCC[index][1] = cY
                index += 1

    orderedTargets = order_targets(pointsCCC)

    # print("Raw Points %s" % pointsCCC)
    # print("Ordered targets: %s" % orderedTargets)

    drawTargetNumbers(img, 0, orderedTargets[0][0], orderedTargets[0][1])

    drawTargetNumbers(img, 1, orderedTargets[1][0], orderedTargets[1][1])

    drawTargetNumbers(img, 2, orderedTargets[2][0], orderedTargets[2][1])

    drawTargetNumbers(img, 3, orderedTargets[3][0], orderedTargets[3][1])

    drawTargetNumbers(img, 4, orderedTargets[4][0], orderedTargets[4][1])

    K = np.array([[531, 0, 320], [0, 531, 240], [0, 0, 1]]).astype(float)

    isPoseFound, rvec, tvec = cv2.solvePnP(pointsObj, pointsCCC, K, distCoeffs=None)

    # print(isPoseFound)

    # Draw coordinate axes onto the image.  Scale the length of the axes
    # according to the size of the model, so that the axes are visible.
    W = np.amax(pointsObj, axis=0) - np.amin(pointsObj, axis=0)
    # Size of model in X,Y,Z
    L = np.linalg.norm(W)
    # Length of the diagonal of the bounding box
    d = L / 5  # This will be the length of the coordinate axes

    pAxes = np.float32([
        [0, 0, 0],  # origin
        [d, 0, 0],  # x axis
        [0, d, 0],  # y axis
        [0, 0, d]  # z axis
    ])
    pImg, J = cv2.projectPoints(
        objectPoints=pAxes,  # Numpy array, size (N,3)
        rvec=rvec, tvec=tvec,
        cameraMatrix=K, distCoeffs=None)

    pImg = pImg.reshape(-1, 2)  # reshape from size (N,1,2) to (N,2)
    cv2.line(img, tuple(np.int32(pImg[0])),
             tuple(np.int32(pImg[1])), (0, 0, 255), 3)  # x (red)

    cv2.line(img, tuple(np.int32(pImg[0])),
             tuple(np.int32(pImg[2])), (0, 255, 0), 3)  # y (green)

    cv2.line(img, tuple(np.int32(pImg[0])),
             tuple(np.int32(pImg[3])), (255, 0, 0), 3)  # z (blue)


    rvecX = str(rvec[0]).strip('[]')
    rvecY = str(rvec[1]).strip('[]')
    rvecZ = str(rvec[2]).strip('[]')

    tvecX = str(tvec[0]).strip('[]')
    tvecY = str(tvec[1]).strip('[]')
    tvecZ = str(tvec[2]).strip('[]')

    cv2.putText(img, ('tvec: ' + '(' + tvecX +', ' + tvecY + ', ' + tvecZ + ')'), org=(0, 420), fontFace=3,

                fontScale=.5, color=(0, 255, 255))

    cv2.putText(img, ('rvec: ' + '(' + rvecX +', ' + rvecY + ', ' + rvecZ + ')'), org=(0, 460), fontFace=3,

                fontScale=.5, color=(0, 255, 255))



    # cv2.putText(img, str(tvec), org=(50, 200), fontFace=3,
    #
    #             fontScale=.5, color=(0, 0, 255))

    cv2.imshow("Image", img)

    cv2.waitKey(0)


def drawTargetNumbers(img, number, x, y):
    cv2.putText(img, str(number), org=(int(x - 7), int(y - 7)), fontFace=3,

                fontScale=.75, color=(0, 0, 255))


if __name__ == '__main__':
    main()
