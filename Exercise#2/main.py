
import cv2 as opencv
import numpy
import matplotlib.pyplot as plt

"""
Bu fonksiyon ile resimdeki position(x,y) noktasından kernel değer kadar window dönme sağlanır 
"""
def getImageWindow(image, kernel, position):
    padSize = int((kernel[0] - 1) / 2)
    image_padding = numpy.pad(image, padSize)
    positionX = position[0] + padSize
    positionY = position[1] + padSize

    positionXStart = positionX-padSize
    positionXEnd = positionX+padSize
    positionYStart = positionY - padSize
    positionYEnd = positionY + padSize

    window = image_padding[
             positionXStart:positionXEnd+1,
             positionYStart:positionYEnd+1]
    return window

"""
Verilen resim ve kernel için box filter uygulanması sağlanır 
"""
def customBoxFilter(image , kernel):
    boxFilterImage = numpy.array(image)
    for row in range(len(image)):
        for col in range(len(image[row])):
            position = (row, col)
            window = getImageWindow(image,kernel,position)
            boxFilterImage[row][col] = round(numpy.sum(window) / (kernel[0] * kernel[1]))

    return boxFilterImage

if __name__ == '__main__':
    # INITIALIZATIN
    image = opencv.imread("test.jpg", opencv.IMREAD_GRAYSCALE)
    image_Arr = numpy.array(image)

    # CUSTOM BOX FILTER 1
    print("CUSTOM 3x3 box filter called")
    kernal_size = (3, 3)
    output_1_1 = customBoxFilter(image_Arr, kernal_size)
    output_2_1 = opencv.boxFilter(image, -1, kernal_size, normalize=True)
    diff_1 = numpy.abs(output_1_1-output_2_1)


    # CUSTOM BOX FILTER 2
    print("CUSTOM 11x11 box filter called")
    kernal_size = (11, 11)
    output_1_2 = customBoxFilter(image_Arr, kernal_size)
    output_2_2 = opencv.boxFilter(image, -1, kernal_size, normalize=True)
    diff_2 = numpy.abs(output_1_2 - output_2_2)


    # CUSTOM BOX FILTER 3
    print("CUSTOM 21x21 box filter called")
    kernal_size = (21, 21)
    output_1_3 = customBoxFilter(image_Arr, kernal_size)
    output_2_3 = opencv.boxFilter(image, -1, kernal_size, normalize=True)
    diff_3 = numpy.abs(output_1_3 - output_2_3)

    opencv.imshow('ORGINAL IMAGE', image)
    opencv.imshow('CUSTOM FILTER 3x3', output_1_1)
    opencv.imshow('BOX FILTER 3x3', output_2_1)
    opencv.imshow('CUSTOM FILTER 11x11', output_1_2)
    opencv.imshow('BOX FILTER 11x11', output_2_2)
    opencv.imshow('CUSTOM FILTER 21x21', output_1_3)
    opencv.imshow('BOX FILTER 21x21', output_2_3)

    opencv.waitKey(0)
    opencv.destroyAllWindows()
