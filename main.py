import cv2 as cv
import numpy as np


def reshape_frame(frame, scale=0.7):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimens = (width, height)
    return cv.resize(frame, dimens, interpolation=cv.INTER_AREA)


img = cv.imread('pixel.jpg')
img_resized = reshape_frame(img, scale=0.3)

cv.imshow('Pixel', img_resized)

b, g, r = cv.split(img_resized)

# RGB channels in grayscale
# cv.imshow('Pixel - blue channel (grayscale)', b)
# cv.imshow('Pixel - green channel (grayscale)', g)
# cv.imshow('Pixel - red channel (grayscale)', r)

# RGB channels

x = np.zeros_like(b)
b = cv.merge([b, x, x])
g = cv.merge([x, g, x])
r = cv.merge([x, x, r])

cv.imshow("Pixel - red channel", r)
cv.imshow("Pixel - green channel", g)
cv.imshow("Pixel - blue channel", b)

cv.imwrite("pixel_red.jpg", r)
cv.imwrite("pixel_green.jpg", g)
cv.imwrite("pixel_blue.jpg", b)

# CMY channels

b, g, r = cv.split(img_resized)
x = np.full_like(b, 255)

c = cv.merge([x, x, r])
m = cv.merge([x, g, x])
y = cv.merge([b, x, x])

cv.imshow("Pixel - cyan channel", c)
cv.imshow("Pixel - magenta channel", m)
cv.imshow("Pixel - yellow channel", y)

cv.imwrite("pixel_cyan.jpg", c)
cv.imwrite("pixel_magenta.jpg", m)
cv.imwrite("pixel_yellow.jpg", y)

cv.waitKey(0)

cv.destroyAllWindows()
