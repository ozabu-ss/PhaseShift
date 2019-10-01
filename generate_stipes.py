import cv2
import numpy as np

try:
    xrange
except NameError:
    xrange = range

WIDTH = 1000
HEIGHT = 1000
STRIPES = 50


def generate_fringes(h, w, n, freq=4):
    alpha = 2. * np.pi / freq  # phase shift by 2*PI / freq
    fringes = list(xrange(freq))

    # period length:
    phi = w / n

    # scale factor delta = 2pi/phi
    delta = 2 * np.pi / phi
    for phase_shift in range(freq):
        fringes[phase_shift] = np.zeros((h, w))
        vertical_line = [(np.cos(x * delta + alpha * phase_shift) + 1) * 120 for x in xrange(w)]
        fringes[phase_shift][:, :] = vertical_line

    return fringes


def generate_horizontal_fringes():
    return map(cv2.transpose, generate_fringes(WIDTH, HEIGHT, STRIPES))


if __name__ == '__main__':
    images = generate_horizontal_fringes()
    for i, img in enumerate(images):
        img = img.astype(np.uint8)
        cv2.imshow('fringe', img)
        cv2.imwrite(f'fringe_{i}.png', img)
        cv2.waitKey(3)
