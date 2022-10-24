#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 15:14:46 2020

script to binarize and crop standardized images of flowers (primula)

@author: Thomsn
"""
#!pip install scikit-image
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters
from skimage.filters.thresholding import _cross_entropy
from skimage import io
from skimage.color import rgb2gray
from skimage.filters import try_all_threshold
from skimage import measure
from pyefd import elliptic_fourier_descriptors
from pyefd import reconstruct_contour
import matplotlib
from skimage.data import page
from skimage.filters import (threshold_otsu, threshold_niblack,
                             threshold_sauvola)
from skimage.morphology import reconstruction



os.chdir('/Users/Thomsn/Desktop/posterscent/morphom/box')
images = os.listdir()
for im in images[:5]:
    moon = io.imread(im)
    new_moon = moon[1300:2700,2300:3700]
    new_moon = rgb2gray(new_moon)
    
    matplotlib.rcParams['font.size'] = 9
    
    
    image = new_moon
    
    
    # window_size = 201
    # thresh_niblack = threshold_niblack(image, window_size=window_size, k=.1)
    
    # window_size = 201
    
    # thresh_sauvola = threshold_sauvola(image, window_size=window_size)
    
    # binary_niblack = image > thresh_niblack
    # #binary_niblack = binary_niblack > threshold_otsu(image)
    
    # binary_sauvola = image > thresh_sauvola
    
    # plt.figure(figsize=(8, 7))
    # plt.subplot(2, 2, 1)
    # plt.imshow(image, cmap=plt.cm.gray)
    # plt.title('Original')
    # plt.axis('off')
    
    # plt.subplot(2, 2, 2)
    # plt.title('Global Threshold')
    # plt.imshow(binary_global, cmap=plt.cm.gray)
    # plt.axis('off')
    
    # plt.subplot(2, 2, 3)
    # plt.imshow(binary_niblack, cmap=plt.cm.gray)
    # plt.title('Niblack Threshold')
    # plt.axis('off')
    
    # plt.subplot(2, 2, 4)
    # plt.imshow(binary_sauvola, cmap=plt.cm.gray)
    # plt.title('Sauvola Threshold')
    # plt.axis('off')
    
    # plt.show()
    
    
    seed = np.copy(image)
    seed[1:-1, 1:-1] = image.max()
    mask = image
    
    filled = reconstruction(seed, mask, method='erosion')
    
    seed = np.copy(image)
    seed[1:-1, 1:-1] = image.min()
    rec = reconstruction(seed, mask, method='dilation')
    
    # fig, ax = plt.subplots(2, 2, figsize=(5, 4), sharex=True, sharey=True)
    # ax = ax.ravel()
    plain = rec - filled
    image = plain
    
    
    binary_global = image > threshold_otsu(image)
    
    
    # ax[0].imshow(image, cmap='gray')
    # ax[0].set_title('Original image')
    # ax[0].axis('off')
    
    # ax[1].imshow(filled, cmap='gray')
    # ax[1].set_title('after filling holes')
    # ax[1].axis('off')
    
    # plt.imshow(-filled+rec, cmap='gray')
    # ax[2].set_title('holes')
    # ax[2].axis('off')
    
    # ax[3].imshow(image-rec, cmap='gray')
    # ax[3].set_title('peaks')
    # ax[3].axis('off')
    # plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    #!pip install pyefd
    ## get shape
    
    
    contours = measure.find_contours(binary_global, 0.8)
    
    
    # fig, ax = plt.subplots()
    # ax.imshow(binary_global, cmap=plt.cm.gray)
    
    # for n, contour in enumerate(contours):
    #     plt.plot(contours[0][:, 1], contours[0][:, 0], linewidth=4)
    
    # ax.axis('image')
    # ax.set_xticks([])
    # ax.set_yticks([])
    # plt.show()
    
    
    coeffs = elliptic_fourier_descriptors(contours[0], order=10)
    
    coeffs.shape[1]
    new_contours = reconstruct_contour(coeffs, locus=(0, 0), num_points=200)
    
    
    fig, ax = plt.subplots()
    ax.imshow(binary_global, cmap=plt.cm.gray)
    
    plt.plot(new_contours[:,1],new_contours[:,0], linewidth=4)
    plt.show
    
    
