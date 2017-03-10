import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

IMAGE = ""
CIRCULARIMAGES = ""

#image must be passed from which circles are detected and retruned
def detectCircles(fromImg):
    fromImg = cv2.medianBlur(fromImg, 5)
    circles = cv2.HoughCircles(fromImg, cv2.cv.CV_HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)
    circles = np.uint16(np.around(circles))
    return circles

#circular region must be passed with image to get cropped image and returns number of images cropped
def cropCircularImage(circles, fromImg):
    j = 0;

    defaultPath = os.getcwd()
    if not os.path.exists(defaultPath + "\\temp"):
        os.makedirs(defaultPath + "\\temp")
    os.chdir(defaultPath + "\\temp")
    
    for i in circles[0,:]:
        x = (i[0] - i[2])
        y = (i[1] - i[2])
        h = y + ((i[1] + i[2]) - (i[1] - i[2]))
        w = x + ((i[0] + i[2]) - (i[0] - i[2]))
        cropImg = fromImg[y:h, x:w]
        j = j + 1
        cv2.imwrite('circular' + str(j) + '.png', cropImg)

    os.chdir(defaultPath)
    return j

#retruns number files present in a directory which is passed to function
def numOfFiles(inDirectory):
    path = inDirectory
    path, dirs, files = os.walk(inDirectory).next()
    fileCount = len(files)
    return fileCount

#pass GRAY IMAGE and returns detected edges in an image
def detectEdgeFrom(fromTemplate):
    detectedEdgeTemplate = cv2.Canny(fromTemplate, 50, 200)
    return detectedEdgeTemplate


def compareTemplate(inImage):
    defaultPath = os.getcwd()
    listClassifier = []
    listClassifier = os.listdir(defaultPath + "\\learning_set")
    for classifier in listClassifier:
        count = numOfFiles(defaultPath + "\\learning_set\\" + classifier)
        for i in range(count):
            temp = defaultPath + "\\learning_set\\" + classifier + "\\" + str(i+1) + ".png"
            print temp
            if templateMatch(IMAGE, temp):
                break
            #templateMatch("road2.png", "road2.png")

def templateMatch(inImage, temp):
    img_rgb = cv2.imread(inImage)
    img_rgb = cv2.medianBlur(img_rgb,3)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    template = cv2.imread(temp,0)
    template = cv2.medianBlur(template,3)
    w, h = template.shape[::-1]

    res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where( res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
        print "match found"
        return True
    return False
    
def detectSign():
    IMAGE = raw_input("ENTER IMAGE: ")
    img = cv2.imread(IMAGE, 0)
    c = detectCircles(img)
    img = cv2.imread(IMAGE)
    numOfFiles = cropCircularImage(c, img)
    defaultPath = os.getcwd()
    for i in range(numOfFiles):
        CIRCULARIMAGES = defaultPath + "\\temp\\circular" + str(i+1) + ".png"
        compareTemplate(CIRCULARIMAGES)

detectSign()
