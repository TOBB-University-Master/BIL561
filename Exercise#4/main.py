
import cv2 as opencv
import numpy
import matplotlib.pyplot as plt
import math
def adaptiveMedianFilter(image, kernelStartSize, kernelMaxSize , centerWeight=0):
    print("Adaptive Median Filter")
    padding_size = math.floor(kernelMaxSize / 2)

    #print(image)
    image_padded = numpy.pad(image, padding_size)
    #print(image_padded)

    output_image = numpy.array(image)
    for i in range(len(image)):
        for j in range(len(image[i])):

            kSize = kernelStartSize
            zxy = image[i][j]
            levelBFlag = False

            # LEVEL A
            # Each iteration kernel windows size should be increase
            while kSize < kernelMaxSize:
                kPadding = math.floor( kSize / 2)
                x1 = i+padding_size-kPadding
                x2 = i+padding_size+kPadding+1
                y1 = j+padding_size-kPadding
                y2 = j+padding_size+kPadding+1
                subset = image_padded[x1:x2, y1:y2]

                subset = subset.flatten()
                for w in range(centerWeight):
                    subset = numpy.append(subset, zxy)

                zmin = numpy.min(subset)
                zmax = numpy.max(subset)
                zmed = numpy.median(subset)

                if zmin < zmed and zmed < zmax:
                    levelBFlag = True
                    break
                else:
                    kSize += 2
                    if kSize >= kernelMaxSize:
                        output_image[i][j] = zmed
            # LEVEL B
            if levelBFlag:
                if zmin < zxy and zxy < zmax:
                    output_image[i][j] = zxy
                else:
                    output_image[i][j] = zmed

            # TODO: Hocaya sorulacak
            # Aşağıdaki durum ile orjinal pseudocode arasında bir fark var mı? Olmaması gerektiği bekleniyor
            # Sınavda filtre uygulanırken padding uygulanması beklenecek mi yoksa padding olmadan iç çercevede mi uygulanacak
            # variance değerinin yüksek olması durumu sorulacak (yani - değerlerin 0 yüksek değerlerin 255e çekilmesi durumu doğru mu yoksa baştan mı map edilmeli değerler )
            """ 
            while kSize < kernelMaxSize:
                kPadding = math.floor( kSize / 2)
                x1 = i+padding_size-kPadding
                x2 = i+padding_size+kPadding+1
                y1 = j+padding_size-kPadding
                y2 = j+padding_size+kPadding+1
                subset = image_padded[x1:x2, y1:y2]

                zmin = numpy.min(subset)
                zmax = numpy.max(subset)
                zmed = numpy.median(subset)
                # LEVEL B
                if zmin < zmed and zmed < zmax:
                    if zmin < zxy and zxy < zmax:
                        output_image[i][j] = zxy
                    else:
                        output_image[i][j] = zmed

                    break
                # LEVEL A
                else:
                    kSize += 2
                    if kSize==kernelMaxSize:
                        output_image[i][j] = zmed
            """

    return output_image


def adaptiveMeanFilter(image, kernel, variance_of_noise):
    print("Adaptive Mean Filter")

    if(type(kernel) is not tuple):
        print("Kernel type should be tuple")
        return image

    padding_size = math.floor(kernel[0] / 2)
    print(image)
    image_padded = numpy.pad(image, padding_size)
    print(image_padded)

    output_image = numpy.array(image)
    for i in range(len(image)):
        for j in range(len(image[i])):
            x1 = i
            x2 = x1+kernel[0]
            y1 = j
            y2 = y1+kernel[1]
            subset = image_padded[x1:x2, y1:y2]
            subset = subset / 255.0

            mean = numpy.mean(subset)
            variance = numpy.var(subset)

            image_x_y = image[i][j] / 255.0
            val = image_x_y - (variance_of_noise / variance)*(image_x_y - mean)

            # TODO: Yazılacak yeni değer float tutmalı & yeni bir array içinde olmalı
            output_image[i][j] = val * 255
            if output_image[i][j] > 255:
                output_image[i][j] = 255
            elif output_image[i][j] < 0:
                output_image[i][j] = 0

    return output_image


def showHistogramAndImage(image, windowName):
    # Histogram
    opencv_hist = opencv.calcHist([image], [0], None, [256], [0, 256])
    plt.plot(opencv_hist)
    plt.title("Histogram " + windowName)
    plt.xlabel('Gri Seviye Değeri')
    plt.ylabel('Piksel Sayısı')
    plt.xlim([0, 256])
    plt.show()

    # Histogram Equalization
    equalized_image = opencv.equalizeHist(image)
    opencv_hist_eq = opencv.calcHist([equalized_image], [0], None, [256], [0, 256])
    plt.plot(opencv_hist_eq)
    plt.title("Histogram Equ. " + windowName)
    plt.xlabel('Gri Seviye Değeri')
    plt.ylabel('Piksel Sayısı')
    plt.xlim([0, 256])
    plt.show()

    opencv.imshow(windowName, image)

if __name__ == '__main__':

    # Image read from file
    clean_image = opencv.imread("cleanImage.jpg", opencv.IMREAD_GRAYSCALE)
    input_image = opencv.imread("GaussianNoise.jpg", opencv.IMREAD_GRAYSCALE)
    #input_image_median = opencv.imread("SaltPepperNoise_1.jpg", opencv.IMREAD_GRAYSCALE)
    input_image_median = opencv.imread("SaltPepperNoise_2.jpg", opencv.IMREAD_GRAYSCALE)

    exercise = 2
    if exercise == 1:
        # Adaptive Mean Filter
        output_1_1 = adaptiveMeanFilter(input_image, (5,5), 0.001225)
        showHistogramAndImage(output_1_1, "ADAPTIVE MEAN #1 ")

        output_1_4 = adaptiveMeanFilter(input_image, (5, 5), 0.009)
        showHistogramAndImage(output_1_4, "ADAPTIVE MEAN #2 ")

        output_1_5 = adaptiveMeanFilter(input_image, (5, 5), 0.002)
        showHistogramAndImage(output_1_5, "ADAPTIVE MEAN #3 ")

        # OPENCV Filters
        output_1_2 = opencv.boxFilter(input_image, -1, (5, 5))
        showHistogramAndImage(output_1_2, "OPENCV BOX FILTER #1 ")

        output_1_3 = opencv.GaussianBlur(input_image, (5, 5), 0)
        showHistogramAndImage(output_1_3, "OPENCV GAUSSIAN FILTER #1 ")

        # PSNR Comparison
        psnr_0 = opencv.PSNR(clean_image, clean_image)
        print("PSNR 0_0 clear image itself" + str(psnr_0))
        psnr_1_1 = opencv.PSNR(clean_image, output_1_1)
        print("PSNR 1_1 " + str(psnr_1_1) + " variance of noise (0.001225)")
        psnr_1_2 = opencv.PSNR(clean_image, output_1_2)
        print("PSNR 1_2 " + str(psnr_1_2))
        psnr_1_3 = opencv.PSNR(clean_image, output_1_3)
        print("PSNR 1_3 " + str(psnr_1_3))
        psnr_1_4 = opencv.PSNR(clean_image, output_1_4)
        print("PSNR 1_4 " + str(psnr_1_4) + " variance of noise (0.009)")
        psnr_1_5 = opencv.PSNR(clean_image, output_1_5)
        print("PSNR 1_5 " + str(psnr_1_5) + " variance of noise (0.002)")

    elif exercise == 2:
        # Adaptive Median Filter
        output_2_1 = adaptiveMedianFilter(input_image_median, 3, 7)
        showHistogramAndImage(output_2_1, "ADAPTIVE MEDIAN #1 ")

        output_2_2 = opencv.medianBlur(input_image_median, 3)
        showHistogramAndImage(output_2_2, "OPENCV MEDIAN 3x3 ")

        output_2_3 = opencv.medianBlur(input_image_median, 5)
        showHistogramAndImage(output_2_3, "OPENCV MEDIAN 5x5 ")

        output_2_4 = opencv.medianBlur(input_image_median, 7)
        showHistogramAndImage(output_2_4, "OPENCV MEDIAN 7x7 ")

        output_2_5 = adaptiveMedianFilter(input_image_median, 3, 7, 3)
        showHistogramAndImage(output_2_5, "ADAPTIVE MEDIAN WEIGHT:3  ")

        output_2_6 = adaptiveMedianFilter(input_image_median, 5, 7, 5)
        showHistogramAndImage(output_2_6, "ADAPTIVE MEDIAN WEIGHT:5  ")

        output_2_7 = adaptiveMedianFilter(input_image_median, 7, 7, 7)
        showHistogramAndImage(output_2_6, "ADAPTIVE MEDIAN WEIGHT:7  ")

        # PSNR Comparison
        psnr_2_1 = opencv.PSNR(clean_image, output_2_1)
        print("PSNR 2_1 " + str(psnr_2_1) + " - ")
        psnr_2_2 = opencv.PSNR(clean_image, output_2_2)
        print("PSNR 2_2 " + str(psnr_2_2) + " - ")
        psnr_2_3 = opencv.PSNR(clean_image, output_2_3)
        print("PSNR 2_3 " + str(psnr_2_3) + " - ")
        psnr_2_4 = opencv.PSNR(clean_image, output_2_4)
        print("PSNR 2_4 " + str(psnr_2_4) + " - ")
        psnr_2_5 = opencv.PSNR(clean_image, output_2_5)
        print("PSNR 2_5 " + str(psnr_2_5) + " - ")
        psnr_2_6 = opencv.PSNR(clean_image, output_2_6)
        print("PSNR 2_6 " + str(psnr_2_6) + " - ")
        psnr_2_7 = opencv.PSNR(clean_image, output_2_7)
        print("PSNR 2_7 " + str(psnr_2_7) + " - ")

    elif exercise == 21:

        showHistogramAndImage(input_image_median, "ORGINAL IMAGE ")

        output_2_1 = adaptiveMedianFilter(input_image_median, 3, 7)
        showHistogramAndImage(output_2_1, "ADAPTIVE MEDIAN #1 ")

        output_2_2 = opencv.medianBlur(input_image_median, 3)
        showHistogramAndImage(output_2_2, "OPENCV MEDIAN 3x3 ")


    # Wait & Close All Windows

    opencv.waitKey(0)
    opencv.destroyAllWindows()

