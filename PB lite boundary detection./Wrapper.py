#!/usr/bin/env python

"""
CMSC733 Spring 2020: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 1 Starter Code


Author(s): 
Nitin J. Sanket (nitin@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park

Chahat Deep Singh (chahat@terpmail.umd.edu) 
PhD Student in Computer Science,
University of Maryland, College Park

Chayan Kumar Patodi (ckp1804@terpmail.umd.edu)
Graduate Student pursuing Masters in Robotics,
University of Maryland, College Park
"""

# Code starts here:

import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os
import glob

# Helper Functions.

def rotateImage(image, angle):
	image_center = tuple(np.array(np.array(image).shape[1::-1]) / 2)
	rotation_matrix = cv2.getRotationMatrix2D(image_center, angle, 1.0)
	result = cv2.warpAffine(image, rotation_matrix, image.shape[1::-1], flags=cv2.INTER_LINEAR)
	return result


def GaussianKernel(n, sigma):
	variance = sigma ** 2
	size = int((n - 1) / 2)
	g = np.asarray([[(-y * np.exp(-1 * (x ** 2 + y ** 2) / (2 * variance))) for x in range(-size, size + 1)] for y in
					range(-size, size + 1)])
	gauss = g / (2 * np.pi * variance * variance)

	return gauss


def Gaussian1D(sigma, mu, x, order):
	x = np.array(x) - mu
	variance = sigma ** 2
	gauss = (1 / np.sqrt(2 * np.pi * variance)) * (np.exp((-1 * x * x) / (2 * variance)))
	if order == 0:
		return gauss
	elif order == 1:
		gauss = - gauss * ((x) / (variance))
		return gauss
	else:
		gauss = gauss * (((x * x) - variance) / (variance ** 2))
		return gauss


def Gaussian2D(k, sigma):
	size = int((k - 1) / 2)
	variance = sigma ** 2
	s = np.asarray([[x ** 2 + y ** 2 for x in range(-size, size + 1)] for y in range(-size, size + 1)])
	Gauss = (1 / np.sqrt(2 * np.pi * variance)) * np.exp(-s / (2 * variance))
	return Gauss


def LOG2D(k, sigma):
	size = int((k - 1) / 2)
	variance = sigma ** 2
	s = np.asarray([[x ** 2 + y ** 2 for x in range(-size, size + 1)] for y in range(-size, size + 1)])
	p = (1 / np.sqrt(2 * np.pi * variance)) * np.exp(-s / (2 * variance))
	Laplacian = p * (s - variance) / (variance ** 2)
	return Laplacian


def makefilter(scale, PhaseX, PhaseY, Points, k):
	gx = Gaussian1D(3 * scale, 0, Points[0, ...], PhaseX)
	gy = Gaussian1D(scale, 0, Points[1, ...], PhaseY)
	image = gx * gy
	image = np.reshape(image, (k, k))
	return image

def binary(image, bins):
	binary_img = image * 0
	for r in range(0, image.shape[0]):
		for c in range(0, image.shape[1]):
			if image[r, c] == bins:
				binary_img[r, c] = 1
			else:
				binary_img[r, c] = 0

	return binary_img

def compute_gradient(maps, numbins, mask_1, mask_2):
	maps = maps.astype(np.float64)
	gradient = np.zeros((maps.shape[0], maps.shape[1], 12))
	for m in range(0, 12):
		chi_square = np.zeros((maps.shape))
		for i in range(1, numbins):
			tmp = binary(maps, i)
			g_i = cv2.filter2D(tmp, -1, mask_1[m])
			h_i = cv2.filter2D(tmp, -1, mask_2[m])
			chi_square = chi_square + ((g_i - h_i) ** 2) / (g_i + h_i + 0.0001)
		gradient[:, :, m] = chi_square

	return gradient

#Half-Disk.

def half_disk(radius):
	halfd_ = np.zeros((radius * 2, radius * 2))
	rad_ = radius ** 2;
	for i in range(0, radius):
		m = (i - radius) ** 2
		for j in range(0, 2 * radius):
			if m + (j - radius) ** 2 < rad_:
				halfd_[i, j] = 1
	return halfd_

#Filters.

def Oriented_DoG():
	sigma = [1, 3]
	orients = 16
	orientation = np.arange(0, 360, 360 / orients)
	plt.figure(figsize=(25, 5))
	val = []

	for i in range(0, len(sigma)):

		kernel = (GaussianKernel(7, sigma[i]))
		for j in range(0, orients):
			filterg = rotateImage(kernel, orientation[j])
			val.append(filterg)
			plt.suptitle("OrientedDoG")
			plt.subplot(len(sigma), orients, orients * (i) + j + 1)
			plt.axis('off')
			plt.imshow(val[orients * (i) + j], cmap='gray')
	plt.show()

	return val

def LML():
	k = 49
	scaleX = np.sqrt(2) ** np.array([1, 2, 3])
	Orientation = 6
	Rotation_ = 12
	Bar_ = len(scaleX) * Orientation
	Edge_ = len(scaleX) * Orientation
	nF = Bar_ + Edge_ + Rotation_
	F = np.zeros([k, k, nF])
	hK = (k - 1) / 2

	x = [np.arange(-hK, hK + 1)]
	y = [np.arange(-hK, hK + 1)]
	[x, y] = np.meshgrid(x, y)
	orgPts = [x.flatten(), y.flatten()]
	orgPts = np.array(orgPts)

	count = 0

	for scale in range(len(scaleX)):
		for orient in range(Orientation):
			angle = (np.pi * orient) / Orientation
			cosine_ = np.cos(angle)
			sin_ = np.sin(angle)
			rotPts = [[cosine_, -sin_], [sin_, cosine_]]
			rotPts = np.array(rotPts)
			rotPts = np.dot(rotPts, orgPts)
			# print(rotPts)
			F[:, :, count] = makefilter(scaleX[scale], 0, 1, rotPts, k)
			F[:, :, count + Edge_] = makefilter(scaleX[scale], 0, 2, rotPts, k)
			count = count + 1

	count = Bar_ + Edge_
	scales = np.sqrt(2) ** np.array([1, 2, 3, 4])

	for i in range(len(scales)):
		F[:, :, count] = Gaussian2D(k, scales[i])
		count = count + 1

	for i in range(len(scales)):
		F[:, :, count] = LOG2D(k, scales[i])
		count = count + 1

	for i in range(len(scales)):
		F[:, :, count] = LOG2D(k, 3 * scales[i])
		count = count + 1

	plt.figure(figsize=(12, 8))
	for i in range(0, 48):
		plt.subplot(6, 8, i + 1)
		plt.axis('off')
		plt.imshow(F[:, :, i], cmap='gray')
		plt.suptitle("LML")
	plt.show()

	return F

def LMS():
	k = 49
	scaleX = np.sqrt(2) ** np.array([0, 1, 2])
	Orientation = 6
	Rotation_ = 12
	Bar_ = len(scaleX) * Orientation
	Edge_ = len(scaleX) * Orientation
	nF = Bar_ + Edge_ + Rotation_
	F = np.zeros([k, k, nF])
	hK = (k - 1) / 2

	x = [np.arange(-hK, hK + 1)]
	y = [np.arange(-hK, hK + 1)]
	[x, y] = np.meshgrid(x, y)
	orgPts = [x.flatten(), y.flatten()]
	orgPts = np.array(orgPts)

	count = 0

	for scale in range(len(scaleX)):
		for orient in range(Orientation):
			angle = (np.pi * orient) / Orientation
			cosine_ = np.cos(angle)
			sin_ = np.sin(angle)
			rotPts = [[cosine_, -sin_], [sin_, cosine_]]
			rotPts = np.array(rotPts)
			rotPts = np.dot(rotPts, orgPts)
			# print(rotPts)
			F[:, :, count] = makefilter(scaleX[scale], 0, 1, rotPts, k)
			F[:, :, count + Edge_] = makefilter(scaleX[scale], 0, 2, rotPts, k)
			count = count + 1

	count = Bar_ + Edge_
	scales = np.sqrt(2) ** np.array([0, 1, 2, 3])

	for i in range(len(scales)):
		F[:, :, count] = Gaussian2D(k, scales[i])
		count = count + 1

	for i in range(len(scales)):
		F[:, :, count] = LOG2D(k, scales[i])
		count = count + 1

	for i in range(len(scales)):
		F[:, :, count] = LOG2D(k, 3 * scales[i])
		count = count + 1

	plt.figure(figsize=(12, 8))
	for i in range(0, 48):
		plt.subplot(6, 8, i + 1)
		plt.axis('off')
		plt.imshow(F[:, :, i], cmap='gray')
		plt.suptitle("LMS")
	plt.show()

	return F

def gabor(sigma, theta, lambda_, psi, gamma):
	gabor_ = list()
	filters = 15
	for k in sigma:
		xsigma = k
		ysigma = float(k) / gamma

		std_ = 3
		xmax = np.ceil(max(1, max(abs(std_ * xsigma * np.cos(theta)), abs(std_ * ysigma * np.sin(theta)))))
		ymax = np.ceil(max(1, max(abs(std_ * xsigma * np.sin(theta)), abs(std_ * ysigma * np.cos(theta)))))
		xmin = -xmax
		ymin = -ymax
		(y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))

		x_theta = x * np.cos(theta) + y * np.sin(theta)
		y_theta = -x * np.sin(theta) + y * np.cos(theta)

		gab_ = np.exp(-.5 * (x_theta ** 2 / xsigma ** 2 + y_theta ** 2 / ysigma ** 2)) * np.cos(
			2 * np.pi / lambda_ * x_theta + psi)
		angle = np.linspace(0, 360, filters)
		for i in range(filters):
			image = rotateImage(gab_, angle[i])
			gabor_.append(image)

		l_ = len(gabor_)
		for i in range(l_):
			plt.subplot(l_ / 5, 5, i + 1)
			plt.axis('off')
			plt.imshow(gabor_[i], cmap='gray')
	plt.show()

	return gabor_

def main():

	"""
	Generate Difference of Gaussian Filter Bank: (DoG)
	Display all the filters in this filter bank and save image as DoG.png,
	use command "cv2.imwrite(...)"
	"""

	DoG = Oriented_DoG()

	"""
	Generate Leung-Malik Filter Bank: (LM)
	Display all the filters in this filter bank and save image as LM.png,
	use command "cv2.imwrite(...)"
	"""

	L1 = LML()
	L2 = LMS()

	"""
	Generate Gabor Filter Bank: (Gabor)
	Display all the filters in this filter bank and save image as Gabor.png,
	use command "cv2.imwrite(...)"
	"""

	G = gabor(sigma=[9, 13], theta=0.25, lambda_=7, psi=0.5, gamma=1)

	"""
	Generate Half-disk masks
	Display all the Half-disk masks and save image as HDMasks.png,
	use command "cv2.imwrite(...)"
	"""

	orientation = np.arange(0, 360, 360 / 8)
	scales = np.asarray([5, 7, 10])
	mask_3 = []
	mask_4 = []
	sz = scales.size
	oz = orientation.size

	for i in range(0, sz):
		halfd_ = half_disk(scales[i])
		for m in range(0, oz):
			mask_1 = rotateImage(halfd_, orientation[m])
			mask_3.append(mask_1)
			mask_2 = rotateImage(mask_1, 180)
			mask_4.append(mask_2)
			plt.subplot(sz * 2, oz, oz * 2 * (i) + m + 1)
			plt.axis('off')
			plt.imshow(mask_1, cmap='gray')
			plt.subplot(sz * 2, oz, oz * 2 * (i) + m + 1 + oz)
			plt.axis('off')
			plt.imshow(mask_2, cmap='gray')
	plt.show()

	# Filter_Bank.

	filter_bank = []
	for i in range(0, len(DoG)):
		filter_bank.append(DoG[i])

	for i in range(0, 48):
		filter_bank.append(L1[:, :, i])

	for i in range(0, 48):
		filter_bank.append(L2[:, :, i])

	for i in range(len(G)):
		filter_bank.append(G[i])


	os.chdir("../BSDS500/Images")

	Images_ = []
	for img in sorted(glob.glob("*.jpg")):
		img_ = cv2.imread(img)
		Images_.append(img_)

	Image_No = 7 #Will range from 0-9.
	plt.imshow(cv2.cvtColor(Images_[Image_No], cv2.COLOR_BGR2RGB))
	plt.show()
	os.chdir("../../Code")



	Img_ = cv2.cvtColor(Images_[Image_No], cv2.COLOR_BGR2GRAY)
	Img_C = Images_[Image_No]

	"""
	Generate Texton Map
	Filter image using oriented gaussian filter bank
	"""

	data = np.zeros((Img_.size, len(filter_bank)))
	for i in range(0, len(filter_bank)):
		temp_image = cv2.filter2D(Img_, -1, filter_bank[i])
		temp_image = temp_image.reshape((1, Img_.size))
		data[:, i] = temp_image


	"""
	Generate texture ID's using K-means clustering
	Display texton map and save image as TextonMap_ImageName.png,
	use command "cv2.imwrite('...)"
	"""

	k_means_texton = KMeans(n_clusters=64, n_init=4)
	k_means_texton.fit(data)
	labels = k_means_texton.labels_
	texton_map = np.reshape(labels, (Img_.shape))

	plt.imshow(texton_map, cmap=None)
	plt.title("TextonMap")
	plt.axis('off')
	plt.show()

	"""
	Generate Texton Gradient (Tg)
	Perform Chi-square calculation on Texton Map
	Display Tg and save image as Tg_ImageName.png,
	use command "cv2.imwrite(...)"
	"""

	texton_gradient = compute_gradient(texton_map, 64, mask_1, mask_2)
	tex_gm = np.mean(texton_gradient, axis=2)
	plt.imshow(tex_gm, cmap=None)
	plt.title("TextonGradient")
	plt.axis('off')
	plt.show()

	"""
	Generate Brightness Map
	Perform brightness binning 
	"""
	m = Img_.reshape((Img_.shape[0] * Img_.shape[1]), 1)
	k_means_Brightness = KMeans(n_clusters=16, random_state=4)
	k_means_Brightness.fit(m)
	labels = k_means_Brightness.labels_
	brightness_map = np.reshape(labels, (Img_.shape[0], Img_.shape[1]))
	mini_ = np.min(brightness_map)
	maxx_ = np.max(brightness_map)
	brightnessmap_final = 255 * (brightness_map - mini_) / np.float((maxx_ - mini_))

	plt.imshow(brightnessmap_final, cmap='gray')
	plt.title("BrightnessMap")
	plt.axis('off')
	plt.show()

	"""
	Generate Brightness Gradient (Bg)
	Perform Chi-square calculation on Brightness Map
	Display Bg and save image as Bg_ImageName.png,
	use command "cv2.imwrite(...)"
	"""
	bright_gradient = compute_gradient(brightness_map, 16, mask_1, mask_2)
	bright_gm = np.mean(bright_gradient, axis=2)
	plt.imshow(bright_gm, cmap='gray')
	plt.title("BrightnessGradient")
	plt.axis('off')
	plt.show()

	"""
	Generate Color Map
	Perform color binning or clustering
	"""
	m = Img_C.reshape((Img_C.shape[0] * Img_C.shape[1]), 3)
	k_means_color = KMeans(n_clusters=16, random_state=4)
	k_means_color.fit(m)
	labels = k_means_color.labels_
	colormap = np.reshape(labels, (Img_.shape[0], Img_.shape[1]))
	plt.imshow(colormap)
	plt.title("ColorMap")
	plt.axis('off')
	plt.show()

	"""
	Generate Color Gradient (Cg)
	Perform Chi-square calculation on Color Map
	Display Cg and save image as Cg_ImageName.png,
	use command "cv2.imwrite(...)"
	"""

	color_gradient = compute_gradient(colormap, 16, mask_1, mask_2)
	color_gm = np.mean(color_gradient, axis=2)
	plt.imshow(color_gm)
	plt.title("ColorGradient")
	plt.axis('off')
	plt.show()

	"""
	Read Sobel Baseline
	use command "cv2.imread(...)"
	"""

	if Image_No == 0:
		Image_No = 1
	elif Image_No == 1:
		Image_No = 10

	os.chdir("../BSDS500/SobelBaseline")
	s = cv2.imread("{}.png".format(Image_No))
	plt.imshow(s)
	plt.axis('off')
	#plt.show()

	"""
	Read Canny Baseline
	use command "cv2.imread(...)"
	"""

	os.chdir("../CannyBaseline")
	c = cv2.imread("{}.png".format(Image_No))
	plt.imshow(c)
	plt.axis('off')
	#plt.show()

	"""
	Combine responses to get pb-lite output
	Display PbLite and save image as PbLite_ImageName.png
	use command "cv2.imwrite(...)"
	"""

	sm = cv2.cvtColor(s, cv2.COLOR_BGR2GRAY)
	cm = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)

	Avg_ = (tex_gm + bright_gm + color_gm) / 3
	pb = Avg_ * (0.5 * cm + 0.5 * sm)

	plt.imshow(pb, cmap="gray")
	plt.axis('off')
	plt.show()
    
if __name__ == '__main__':
    main()
 


