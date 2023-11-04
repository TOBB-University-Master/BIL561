
import cv2 as opencv
import numpy
import math
import matplotlib.pyplot as plt

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

    return output_image


"""
Salt-and-Pepper Noise uygular
"""
def saltAndPapper(image):
    input_image = numpy.array(image)
    green_channel = input_image[:, :, 1]

    # Gürültü yoğunluğunu ayarlayabilirsiniz
    noise_intensity = 0.005  # Örnek: %2'lik gürültü yoğunluğu

    # Rastgele koordinatlar oluşturun
    height, width = green_channel.shape
    num_pixels = int(noise_intensity * height * width)
    noise_coordinates = numpy.random.randint(0, height, num_pixels), numpy.random.randint(0, width, num_pixels)

    # Seçilen koordinatlardaki piksellere rastgele siyah veya beyaz değer atayın
    green_channel[noise_coordinates] = numpy.random.choice([0, 255], num_pixels)

    # Kırmızı ve mavi kanalları alın
    red_channel = input_image[:, :, 2]
    blue_channel = input_image[:, :, 0]

    # Yeni RGB görüntüyü oluşturun
    noisy_image = opencv.merge([blue_channel, green_channel, red_channel])

    return noisy_image


def saltAndPepperFrequency(image_channel):
    count = 0
    height, width = image_channel.shape
    for i in range(height):
        for j in range(width):
            if image_channel[i][j]==0 or image_channel[i][j]==255:
                # 3x3 box içinde kontrol edildiğini kabul et/padding yapma
                if(i>1 and i<height-1 and j>1 and j<width-1):
                    zmed = numpy.median(image_channel[i-1:i+1, j-1:j+1])
                    if zmed!=255 or zmed!=0:
                        count+=1

    return count

def detectNoisyChannel(image):
    red_channel = image[:, :, 2]
    green_channel = image[:, :, 1]
    blue_channel = image[:, :, 0]
    redCount = saltAndPepperFrequency(red_channel)
    greenCount = saltAndPepperFrequency(green_channel)
    blueCount = saltAndPepperFrequency(blue_channel)

    print("Red Channel Noisy Count " + str(redCount))
    print("Green Channel Noisy Count " + str(greenCount))
    print("Blue Channel Noisy Count " + str(blueCount))

    if max(redCount,greenCount,blueCount) == redCount:
        print("RED CHANNEL HAS NOISY")
        return "RED"

    elif max(redCount,greenCount,blueCount) == greenCount:
        print("GREEN CHANNEL HAS NOISY")
        return "GREEN"

    elif max(redCount,greenCount,blueCount) == blueCount:
        print("BLUE CHANNEL HAS NOISY")
        return "BLUE"

    return ""

if __name__ == '__main__':

    # TODO: Eğer resim kaydedilmişse muhtemelen kaydetmeden dolayı gürültü diğer kanallara taşıyor
    noisy_image = opencv.imread("lena_rgb_green_channel_noise.jpg")
    input_image = opencv.imread("lena_rgb.png")

    # Tek kanala salt & pepper uygulanır
    noisy_image_2 = saltAndPapper(input_image)
    opencv.imshow("Noise Image With Salt&Pepper", noisy_image_2)

    # Kanallar ayrılır
    red_channel = noisy_image_2[:, :, 2]
    green_channel = noisy_image_2[:, :, 1]
    blue_channel = noisy_image_2[:, :, 0]

    opencv.imshow("Red Channel", red_channel)
    opencv.imshow("Green  Channel", green_channel)
    opencv.imshow("Blue Channel", blue_channel)

    # TRY TO FIND NOISY CHANNEL
    print("TRY TO Find Noisy Channel")
    detectNoisyChannel(noisy_image_2)

    # Problemli kanala Adaptive median filtresi uygulanır
    green_channel_filtered = adaptiveMedianFilter(green_channel, 3,5)
    opencv.imshow("Adaptive Median Filter Green Channel Result", green_channel_filtered)

    # Tüm kanallar birleştirilie
    output = opencv.merge([blue_channel, green_channel_filtered, red_channel])

    opencv.imshow("Orjinal Resim", input_image)
    opencv.imshow("Noise Resim Final", output)

    opencv.waitKey(0)
    opencv.destroyAllWindows()