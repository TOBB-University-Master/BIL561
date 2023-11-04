
import cv2 as opencv
import numpy
import matplotlib.pyplot as plt
import math

def histogram(imageArray):
    hist = [0] * 256
    for rows in imageArray:
        for cols in rows:
            hist[cols] = hist[cols] + 1
    return hist;

def histogramIntensity(histogram, imageMN):
    histIntensity = [0] * 256
    for i in range(len(histogram)):
        histIntensity[i] = histogram[i] / imageMN
    return histIntensity

def histogramEqualization(histogramIntesity):
    histEq = [0] * 256
    for i in range(len(histogramIntesity)):
        if ( i>0 ):
            histEq[i] = (histEq[i-1] + histogramIntesity[i])
        else:
            histEq[i] = histogramIntesity[i]
    for i in range(len(histEq)):
        histEq[i] = histEq[i] * 255
    return histEq

def finalStep(histogramEqualization):
    hist = [0] * 256
    for i in range(len(histogramEqualization)):
        hist[i] = math.ceil(histogramEqualization[i]-0.5)
        hist[i] = round(histogramEqualization[i])
    return hist

def rescanImage( imageArray ,histogramEqualization):
    finalImage = numpy.array(imageArray)
    for rows in range(len(imageArray)):
        for cols in range(len(imageArray[rows])):
            finalImage[rows][cols] = histogramEqualization[imageArray[rows][cols]]
    return finalImage

def HMin(histogramIntesity):
    minIntensity=100
    for i in range(len(histogramIntesity)):
        if(histogramIntesity[i]>1 and histogramIntesity[i]<minIntensity ):
            minIntensity = histogramIntesity[i]
    return minIntensity

def rescanImageAlternative(imageGrayScale):
    IMAGE_SIZE_MN = imageGrayScale.size
    GRAY_LEVEL = 256
    imageHistogram = histogram(imageGrayScale)
    imageIntensity = imageHistogram
    # H[p] = H[p] + 1
    for i in range(len(imageIntensity)):
        if(imageIntensity[i] > 0):
            imageIntensity[i] = imageIntensity[i] + 1

    minVal = HMin(imageIntensity)
    histEq = [0] * GRAY_LEVEL
    for i in range(len(imageIntensity)):
        if (i > 0):
            histEq[i] = imageIntensity[i] + histEq[i-1]
        else:
            histEq[i] = imageIntensity[i]

    for i in range(len(histEq)):
        histEq[i] = round(((histEq[i] - minVal) / (IMAGE_SIZE_MN - minVal)) * (GRAY_LEVEL-1))

    finalImage = numpy.array(imageGrayScale).astype(numpy.uint8)
    for rows in range(len(imageGrayScale)):
        for cols in range(len(imageGrayScale[rows])):
            finalImage[rows][cols] = histEq[imageGrayScale[rows][cols]]
    return finalImage


if __name__ == '__main__':
    imageOrjinal = opencv.imread('test.jpg')
    imageGrayScale = opencv.imread('test.jpg', opencv.IMREAD_GRAYSCALE)

    # OPENCV HISTOGRAM EQUALIZATION
    equalized_image = opencv.equalizeHist(imageGrayScale)
    histEq = opencv.calcHist([equalized_image], [0], None, [256], [0, 256])
    plt.plot(histEq)
    plt.title('OpenCV Histogram Equalization')
    plt.show()
    opencv.imshow('OpenCV Equalization', equalized_image)


    # CUSTOM HISTOGRAM EQUALIZATION
    imageHist = histogram(imageGrayScale)
    imageHistInt = histogramIntensity(imageHist,imageGrayScale.size)
    imageHistEq = histogramEqualization(imageHistInt)
    histogramCeil = finalStep(imageHistEq)
    histogramEqualizationImage = rescanImage(imageGrayScale, histogramCeil)
    hist1 = opencv.calcHist([histogramEqualizationImage], [0], None, [256], [0, 256])
    plt.plot(hist1)
    plt.title('Custom Histogram Equalization #1')
    plt.show()
    opencv.imshow('Custom Histogram Equalization #1', histogramEqualizationImage)


    # HISTOGRAM EQUALIZATION ALTERNATIVE
    rescanImageAlternative = rescanImageAlternative(imageGrayScale)
    hist4 = opencv.calcHist([rescanImageAlternative], [0], None, [256], [0, 256])
    plt.plot(hist4)
    plt.title('Custom Histogram Equalization #2')
    plt.show()
    opencv.imshow('Custom Histogram Equalization #2', rescanImageAlternative)


    # ORGINAL IMAGE
    hist2 = opencv.calcHist([imageGrayScale], [0], None, [256], [0, 256])
    plt.plot(hist2)
    plt.title('Gray Scale')
    plt.show()
    opencv.imshow('ORG. GRAYSCALE IMAGE', imageGrayScale)

    opencv.waitKey(0)
    opencv.destroyAllWindows()
