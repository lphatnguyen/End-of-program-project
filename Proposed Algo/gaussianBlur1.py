from PIL import Image
import math
import numpy as np

# img = Image.open('input.jpg').convert('L')
# r = radiuss
def gauss_blur(img, r):
    imgData = list(img.getdata())
    bluredImg = Image.new(img.mode, img.size)
    bluredImgData = list(bluredImg.getdata())

    rs = int(math.ceil(r * 2.57))

    for i in range(0, img.height):
        for j in range(0, img.width):
            val = 0
            wsum = 0
            for iy in range(i - rs, i + rs + 1):
                for ix in range(j - rs, j + rs + 1):
                    x = min(img.width - 1, max(0, ix))
                    y = min(img.height - 1, max(0, iy))
                    dsq = (ix - j) * (ix - j) + (iy - i) * (iy - i)
                    weight = math.exp(-dsq / (2 * r * r)) / (math.pi * 2 * r * r)
                    val += imgData[y * img.width + x] * weight
                    wsum += weight 
            bluredImgData[i * img.width + j] = round(val / wsum)

    bluredImg.putdata(bluredImgData)
    return bluredImg

image = Image.open("F:/Projet fin d'Etudes/Image/IMG_0493.jpg")
sigma = 5
output = gauss_blur(image, sigma)
output.show()
