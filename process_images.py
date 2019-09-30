import numpy as np
import cv2
import matplotlib.pyplot as plt

def display_plot(img, spectr, img2):
	magnitude_spectrum = 20*np.log(cv2.magnitude(spectr[:,:,0],spectr[:,:,1]))
	plt.subplot(131),plt.imshow(img, cmap = 'gray')
	plt.title('Input Image'), plt.xticks([]), plt.yticks([])
	plt.subplot(132),plt.imshow(magnitude_spectrum, cmap = 'gray')
	plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
	plt.subplot(133),plt.imshow(img2, cmap = 'gray')
	plt.title('Output Image'), plt.xticks([]), plt.yticks([])
	fig, ax = plt.subplots()
	ax.plot(range(magnitude_spectrum.shape[0]), magnitude_spectrum[:][500])
	ax.grid()
	plt.show()

def lpf(img):
	rows, cols = img.shape
	crow, ccol = rows/2, cols/2
	mask = np.zeros((rows, cols, 2), np.uint8)
	mask[crow-100:crow+100, ccol-100:ccol+100] = 1
	dft = cv2.dft(gray, flags = cv2.DFT_COMPLEX_OUTPUT)
	dft_shift = np.fft.fftshift(dft)
	fshift = dft_shift*mask
	f_ishift = np.fft.ifftshift(fshift)
	img_back = cv2.idft(f_ishift)
	img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
	img_back = (img_back - np.min(img_back))/(np.max(img_back) - np.min(img_back))
	#display_plot(img, fshift, img_back)
	return img_back

def hpf(img):
	rows, cols = img.shape
	crow, ccol = rows/2, cols/2
	mask = np.ones((rows, cols, 2), np.uint8)
	mask[crow-50:crow+50, ccol-50:ccol+50] = 0
	dft = cv2.dft(img, flags = cv2.DFT_COMPLEX_OUTPUT)
	dft_shift = np.fft.fftshift(dft)
	fshift = dft_shift*mask
	f_ishift = np.fft.ifftshift(fshift)
	img_back = cv2.idft(f_ishift)
	img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
	img_back = (img_back - np.min(img_back))/(np.max(img_back) - np.min(img_back))
	#display_plot(img, fshift, img_back)
	return img_back

images_files = ['phase_f1.png', 'phase_f2.png', 'phase_f3.png', 'phase_f4.png']
ref_files = ['ref_f1.png', 'ref_f2.png', 'ref_f3.png', 'ref_f4.png']

#read images of the 3D object and filter with high-pass filter
images = []
for filename in images_files:
	img = cv2.imread(filename)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	gray = np.asarray(gray, dtype="float32") / 255.0
	img = hpf(gray)
	images.append(img)

#read reference images and filter with high-pass filter
refs = []
for filename in ref_files:
	ref = cv2.imread(filename)
	gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
	gray = np.asarray(gray, dtype="float32") / 255.0
	refs.append(gray)

#remove DC term from reference images
a = sum(refs) / 4
refs_h = []
for ref in refs:
	refs_h.append(ref - a)

#mutiply reference images with images of the 3D object
images_h = []
for i in range(len(images)):
	images_h.append(np.multiply(images[i],refs_h[i]))

#filter with low-pass filter
filter_img = []
for img in images_h:
	img = lpf(img)
	filter_img.append(img)

#compute wrapped phase
phi = np.arctan((filter_img[3] - filter_img[1]) / (filter_img[0] - filter_img[2]))

cv2.imshow('phi', phi)
cv2.waitKey(0)