
import cv2 as opencv
import numpy
import time

def calculateFilter(image, size, position):
    paddingSize = int((size-1)/2)
    imageWithPadding = numpy.pad(image, paddingSize, 'edge')
    # padding sonucu position olması gereken yer
    positionX = position[0] + paddingSize
    positionY = position[1] + paddingSize
    # position değerine göre window başlangıç ve bitiş noktaları hesaplama
    windowXStart = positionX - paddingSize
    windowXEnd = positionX + paddingSize
    windowYStart = positionY - paddingSize
    windowYEnd = positionY + paddingSize
    window = imageWithPadding[
                    windowXStart:windowXEnd + 1,
                    windowYStart:windowYEnd + 1]

    medianValue = int(numpy.median(window))
    return medianValue


"""
1. Resim üzerinde pixel pixel ilerlenir
2. Her nokta için window çıkarılarak window median değeri hesaplanır
3. En büyük sorunu her pixel için padding işlemini tekrarlaması
"""
def customMedianFilter(image, size):
    output_image = numpy.array(image)
    for row in range(len(image)):
        for col in range(len(image[row])):
            position = (row, col)
            output_image[row][col] = calculateFilter(image, size, position)
    return output_image


"""
1. Bir kez padding yapılır 
2. padding position üzerinden ilerlenir 
"""
def customMedianFilterNew(image, size, isWeighted):
    paddingSize = int((size - 1) / 2)
    imageWithPadding = numpy.pad(image, paddingSize, 'edge')

    output_image = numpy.array(image)
    for row in range(paddingSize,len(imageWithPadding)-paddingSize):
        for col in range(paddingSize,len(imageWithPadding[row])-paddingSize):
            window = imageWithPadding[
                     row-paddingSize:row+paddingSize + 1,
                     col-paddingSize:col+paddingSize + 1]
            if(isWeighted):
                med = window[paddingSize][paddingSize]
                window = window.flatten()
                window = numpy.append(window,[med,med,med])

            output_image[row-paddingSize][col-paddingSize] = int(numpy.median(window))

    return output_image


if __name__ == '__main__':

    psnrTxt = "PSNR {psnr:.2f} - {filterName}"
    orginal_image = opencv.imread("original.jpg", opencv.IMREAD_GRAYSCALE)
    imageArr = []
    for i in range(1,5):
        imageName = "noisyImage_SaltPepper_"+str(i)
        image = opencv.imread(imageName+".jpg", opencv.IMREAD_GRAYSCALE)
        imageArr.append(image)

        print("\n --- IMAGE PROCESS --- " + imageName)

        # CUSTOM MEDIAN NEW FILTER WEIGHTED ~3sec
        t0 = time.time()
        ksize = 7
        output_custom_median_w = customMedianFilterNew(image, ksize, True)
        opencv.imshow(imageName + " CUSTOM MEDIAN WEIGHTED", output_custom_median_w)
        t1 = time.time()
        t2 = t1 - t0
        print("Time " + str(t2))
        output_psnr = opencv.PSNR(orginal_image, output_custom_median_w)
        print(psnrTxt.format(psnr=output_psnr, filterName='CUSTOM MEDIAN WEIGHTED'))

        # CUSTOM MEDIAN NEW FILTER ~3sec
        ksize = 7
        output_custom_median = customMedianFilterNew(image, ksize, False)
        output_psnr = opencv.PSNR(orginal_image, output_custom_median)
        print(psnrTxt.format(psnr=output_psnr, filterName='CUSTOM MEDIAN'))

        # OPENCV MEDIAN FILTER ~0.01sec
        output_median = opencv.medianBlur(image, ksize)
        output_psnr = opencv.PSNR(orginal_image, output_median)
        print(psnrTxt.format(psnr=output_psnr, filterName='OPENCV MEDIAN'))

        # BOX FILTER
        output_box = opencv.boxFilter(image,0,(5,5))
        output_psnr = opencv.PSNR(orginal_image, output_box)
        print(psnrTxt.format(psnr=output_psnr, filterName='OPENCV BOX'))

        # GAUSSIAN
        output_gausian = opencv.GaussianBlur(image, (5,5), 1)
        output_psnr = opencv.PSNR(orginal_image, output_gausian)
        print(psnrTxt.format(psnr=output_psnr, filterName='OPENCV GAUSSIAN'))

        # GAUSSIAN
        output_gausian = opencv.GaussianBlur(image, (7, 7), 1)
        output_psnr = opencv.PSNR(orginal_image, output_gausian)
        print(psnrTxt.format(psnr=output_psnr, filterName='OPENCV GAUSSIAN'))

    opencv.waitKey(0)
    opencv.destroyAllWindows()

