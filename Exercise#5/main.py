
import cv2 as opencv
import numpy
import math

def integral(image):
    print("Calculate integral...")

    # Create an array with size input array
    integral_image = numpy.zeros((image.shape[0],image.shape[1]))

    # Add row
    row = numpy.zeros((1, integral_image.shape[1]))
    integral_image = numpy.vstack((row,integral_image))

    # Add column
    col = numpy.zeros((integral_image.shape[0], 1))
    integral_image = numpy.hstack((col, integral_image))

    # Set array to
    integral_image = numpy.array(integral_image, dtype=numpy.int64)

    # calculate integral
    for i in range(integral_image.shape[0]):
        for j in range(integral_image.shape[1]):
            arrSum = image[:i, :j]
            integral_image[i][j] = arrSum.sum()

    return integral_image


def boxFilter(image, kernelSize):
    print("Box Filter...")

    # calculate image integral
    image_output = numpy.zeros((image.shape[0], image.shape[1]), dtype=numpy.uint8)
    image_integral = integral(image)

    # kernel size kadar padding uygulanır
    kernel_padding_0 = math.floor(kernelSize[0] / 2)
    kernel_padding_1 = math.floor(kernelSize[1] / 2)
    image_integral = numpy.pad(image_integral, (kernel_padding_0, kernel_padding_1),'edge')

    image_integral = opencv.copyMakeBorder(image_integral,0,1,0,1,opencv.BORDER_REFLECT)

    # opencv
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            """
            top_left = (i - kernel_padding_0 - 1, j - kernel_padding_1 - 1)
            top_right = (i - kernel_padding_0 - 1, j + kernel_padding_1 + 1)
            bottom_left = (i + kernel_padding_0 + 1, j - kernel_padding_1 - 1)
            bottom_right = (i + kernel_padding_0 + 1, j + kernel_padding_1 + 1)

            top_left = tuple(x + (kernel_padding_0 + 1) for x in top_left)
            top_right = tuple(x + (kernel_padding_0 + 1) for x in top_right)
            bottom_left = tuple(x + (kernel_padding_1 + 1) for x in bottom_left)
            bottom_right = tuple(x + (kernel_padding_1 + 1) for x in bottom_right)
            """

            # calculate sum of kernel window
            top_left_sum = image_integral[i,j]
            top_right_sum = image_integral[i, j+(kernelSize[1])]
            bottom_left_sum = image_integral[i+(kernelSize[0]), j]
            bottom_right_sum = image_integral[i+(kernelSize[0]), j+(kernelSize[1])]
            kernelSum = (bottom_right_sum - bottom_left_sum - top_right_sum + top_left_sum) / (kernelSize[0] * kernelSize[1])
            image_output[i][j] = kernelSum

    return image_output


if __name__ == '__main__':
    print("*** BIL 561 ***")

    # READ IMAGE
    input_image = opencv.imread("lena_grayscale_hq.jpg", opencv.IMREAD_GRAYSCALE)

    exercise=2
    if 1==exercise:
        # CALCULATE CUSTOM INTEGRAL
        output_1_1 = integral(input_image)
        print("CUSTOM INTEGRAL")
        print(output_1_1)

        output_1_2 = opencv.integral(input_image)
        print("OPENCV INTEGRAL")
        print(output_1_2)

        # Check Diff
        diff = output_1_1 - output_1_2
        print(diff)

        sumdiff = (numpy.array(diff)).sum()
        print(sumdiff)

        # Integral görüntüyü kullanarak belirli bir bölgenin toplamını alabilirsiniz
        x1, y1, x2, y2 = 100, 100, 200, 200
        sum_region = output_1_2[y2, x2] - output_1_2[y1, x2] - output_1_2[y2, x1] + output_1_2[y1, x1]
        print("Belirli bir bölgenin toplamı:", sum_region)

    elif 2==exercise:
        output_2_1 = boxFilter(input_image, (5,5))
        opencv.imshow("aaa", output_2_1)

    input_image = opencv.boxFilter(input_image, -1,(5,5))
    opencv.imshow("bbb", input_image)
    opencv.waitKey(0)
    opencv.destroyAllWindows()
