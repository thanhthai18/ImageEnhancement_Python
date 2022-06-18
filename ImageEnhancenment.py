import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

img_1 = cv2.imread('C:/Users/ADMIN/Desktop/gocTTMT.png', 0) 

### Image Nagative ###
def imageNagative(input):
    output = 255 - input
    
    plt.subplot(131), plt.imshow(input, cmap='gray', vmin=0, vmax=255), plt.title("Input"), plt.axis('off')
    plt.subplot(133), plt.imshow(output, cmap='gray', vmin=0, vmax=255), plt.title("Output"), plt.axis('off')
    plt.show()
    return output





### Log Transformations ###
def logTransformations(input):
    thresh = 1.55
    #tmpImg = np.uint8(np.log1p(input))
    #output = cv2.threshold(tmpImg, thresh, 255, cv2.THRESH_BINARY)[1]
    output = thresh * np.log(1 + input /255)
    

    plt.subplot(131), plt.imshow(input, cmap='gray', vmin=0, vmax=255), plt.title("Input"), plt.axis('off')
    plt.subplot(133), plt.imshow(output, cmap='gray', vmin=0, vmax=255), plt.title("Output"), plt.axis('off')
    plt.show()
    return output






### Power-law Transformations ###
def powerlawTransformations(input, gamma):
    output = np.power(input, gamma) 
    
    plt.subplot(131), plt.imshow(input, cmap='gray', vmin=0, vmax=255), plt.title("Input"), plt.axis('off')
    plt.subplot(133), plt.imshow(output, cmap='gray', vmin=0, vmax=255), plt.title("Output"), plt.axis('off')
    plt.show()
    return output




### Piecewise Linear Transformations ###
def piecewiseLinearTransformations(input, s1, s2):
    height, width = input.shape[:2] 

    rMin = input.min()  #  The minimum value of the gray level of the original image
    rMax = input.max()  #  The maximum gray level of the original image
    #r1, s1 = rMin, 0  # (x1,y1)
    #r2, s2 = rMax, 255  # (x2,y2)
    r1 = rMin
    r2 = rMax

    if s1 < 0 or s1 > 255 or s2 < 0 or s2 > 255 or s1 > s2:
        s1 = 0
        s2 = 255

          
    imgStretch = np.empty((height,width), np.uint8)  #  Create a blank array
    k1 = s1 / r1  # imgGray[h,w] < r1:
    k2 = (s2 - s1) / (r2 - r1)  # r1 <= imgGray[h,w] <= r2
    k3 = (255 - s2) / (255 - r2)  # imgGray[h,w] > r2
    for h in range(height):
        for w in range(width):
           if input[h,w] < r1:
               imgStretch[h,w] = k1 * input[h,w]
           elif r1 <= input[h,w] <= r2:
                   imgStretch[h,w] = k2 * (input[h,w] - r1) + s1
           elif input[h,w] > r2:
                imgStretch[h,w] = k3 * (input[h,w] - r2) + s2

    
    plt.figure(figsize=(10,3.5))
    plt.subplots_adjust(left=0.2, bottom=0.2, right=0.9, top=0.8, wspace=0.1,
    hspace=0.1)
    plt.subplot(131), plt.title("s=T(r)")
    x = [0, r1, r2, 255]
    y = [0, s1, s2, 255]
    plt.plot(x, y)
    plt.axis([0,256,0,256])
    plt.text(105, 25, "(r1,s1)", fontsize=10)
    plt.text(120, 215, "(r2,s2)", fontsize=10)
    plt.xlabel("r, Input value")
    plt.ylabel("s, Output value")
    plt.subplot(132), plt.imshow(input, cmap='gray', vmin=0, vmax=255), plt.title("Input"), plt.axis('off')
    plt.subplot(133), plt.imshow(imgStretch, cmap='gray', vmin=0, vmax=255), plt.title("Output"), plt.axis('off')
    plt.show()
    return imgStretch



### Perspective Transfomation ###
def perspectiveTransfomation(input):
    rows, cols = input.shape

    pt1 = np.float32([[56,65],[368,52],[28,387],[389,290]])
    pt2 = np.float32([[0,0],[300,0],[0,300],[300,300]])

    matrix = cv2.getPerspectiveTransform(pt1, pt2)
    new_img = cv2.warpPerspective(input, matrix, (cols,rows))

    plt.figure(figsize=(10,3.5))
    plt.subplots_adjust()
    plt.subplot(131), plt.imshow(input, cmap='gray', vmin=0, vmax=255), plt.title("Input"), plt.axis('off')
    plt.subplot(133), plt.imshow(new_img, cmap='gray', vmin=0, vmax=255), plt.title("Output"), plt.axis('off')
    plt.show()
    return new_img

### Thresholding Transformations ###
def thresholdingTransformations(input):
    output = cv2.imread('C:/Users/ADMIN/Desktop/gocTTMT.png', 0)
    output[output < 210] = 0
    
    plt.figure(figsize=(10,3.5))
    plt.subplots_adjust()
    plt.subplot(131), plt.imshow(input, cmap='gray', vmin=0, vmax=255), plt.title("Input"), plt.axis('off')
    plt.subplot(133), plt.imshow(output, cmap='gray', vmin=0, vmax=255), plt.title("Output"), plt.axis('off')
    plt.show()

    return output

## Histogram Processing ###


def Hist(input):
    s = input.shape
    H = np.zeros(shape=(256,1))

    for i in range(s[0]):
        for j in range(s[1]):
            k = input[i,j]
            H[k,0] = H[k, 0] + 1
    return H

def showHistograms():
    #img_gray = cv2.cvtColor(input, cv2.COLOR_GRAY2RGB)
    img_gray = img_1
    img_gray = cv2.convertScaleAbs(img_gray, alpha=1.10 , beta= -20)
    s = img_gray.shape
    histg = Hist(img_gray)
    plt.plot(histg)
    x = histg.reshape(1,256)
    y = np.array([])
    y = np.append(y, x[0,0])
    
    for i in range(255):
        k = x[0, i + 1] + y[i]
        y = np.append(y,k)
    y = np.round((y / s[0] * s[1]) * (256 - 1))
    
    #H = Hist(img_1)
    #plt.plot(H)
    plt.show()


## Histogram Equalization ##
def calcGrayHist(I):
    h, w = I.shape[:2]
    grayHist = np.zeros([256], np.uint64)
    for i in range(h):
        for j in range(w):
            grayHist[I[i][j]] += 1
    return grayHist

def equalHist(img):
    h, w = img.shape[0], img.shape[1]
    # Tính độ nạp thẳng độ màu xám
    grayHist = calcGrayHist(img)
    # Tính toán sơ đồ hình vuông thang độ xám tích lũy
    zeroCumuMoment = np.zeros([256], np.uint32)
    for p in range(256):
        if p == 0:
            zeroCumuMoment[p] = grayHist[0]
        else:
            zeroCumuMoment[p] = zeroCumuMoment[p - 1] + grayHist[p]
    # Nhận mối quan hệ ánh xạ giữa mức màu xám đầu vào và mức màu xám đầu ra theo sự tích lũy của thang độ xám
    outPut_q = np.zeros([256], np.uint8)
    cofficient = 256.0 / (h * w)
    for p in range(256):
        q = cofficient * float(zeroCumuMoment[p]) - 1
        if q >= 0:
            outPut_q[p] = np.floor(q)
        else:
            outPut_q[p] = 0
    # Nhận hình ảnh sau khi bản đồ công thức cân bằng
    equalHistImage = np.zeros(img.shape, np.uint8)
    for i in range(h):
        for j in range(w):
            equalHistImage[i][j] = outPut_q[img[i][j]]

    return equalHistImage

def showEqualHist(input):
    equa = equalHist(input)
    plt.imshow(equa)
    plt.show()
    




## Histogram Matching ##

#---------------RUN------------------#


#outputImageNagative = imageNagative(img_1)
#outputLogTransformations = logTransformations(img_1)
#outputPowerlawTransformations = powerlawTransformations(img_1, 4)
#outputPiecewiseLinearTransformations = piecewiseLinearTransformations(img_1, 25, 220)
#outputPerspectiveTransfomation = perspectiveTransfomation(img_1)
#outputThresholdingTransformations = thresholdingTransformations(img_1)
#showHistograms()
#showEqualHist(img_1)
cv2.waitKey(100000)
cv2.destroyAllWindows