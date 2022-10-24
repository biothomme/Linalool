#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 17:10:49 2020

this script is used to analyze and plot morphometrics of primula
for the manuscript with Dötterl, Schaefer et al. - after that an analysis
in R (momocs) needs to be conducted, before plotting can be done with 
shape_pca_part2.py

@author: Thomsn
"""
# imports
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from pyefd import reconstruct_contour
from sklearn.decomposition import PCA
from sklearn import datasets



# functions
# retrieving np-array with shape for one image
def shapeyclean(shapeframe):
    shapearray = []
    for _, row in shapeframe.iterrows():
        coords = [row['x'], row['y']]
        shapearray.append(coords)
    return np.array(shapearray)

# get efd (elliptical fourier descriptors) from shape 
def efdala(shapearray, num_points, num_ft):
    from pyefd import elliptic_fourier_descriptors
    from pyefd import reconstruct_contour
    coeffs = elliptic_fourier_descriptors(shapearray, order=num_ft, normalize=True)
    contours = reconstruct_contour(coeffs, locus=(0, 0), num_points=num_points)
    return coeffs, contours

# plot the newly obtained contour from efd
def efd_plot(contour):
    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111)
    ax.plot(contour[:,1],contour[:,0], linewidth=2)
    return fig.show

# summarizing all coefficients in a dataframe
def coeffary(shapeframe, num_ft):
    coeff_level = [[str(i)]*4 for i in range(1,num_ft+1)]
    coeff_level = np.array([cf2 for cf in coeff_level for cf2 in cf])
    coeff_type = np.array(['a','b','c','d']*num_ft)
    
    header = pd.MultiIndex.from_arrays([coeff_level, coeff_type], names=('coeff_level', 'coeff_type'))
    coefframe = pd.DataFrame(columns = header)
    contframe = pd.DataFrame(columns = range(500))
    for img in set(sh_fr['image']):
        df = sh_fr.loc[sh_fr['image'] == img,:]
        shape = shapeyclean(df)
        efd_coef, efd_cont = efdala(shape, 500, num_ft)
        # fig = efd_plot(efd_cont)
        all_coefs = [ec2 for ec in list(efd_coef) for ec2 in ec]
        with open(f'{img.replace(".JPG",".txt")}', 'w') as sh_file:
            for row in efd_cont:
                row = [str(x) for x in row]
                row = '\t'.join(row) + '\n'
                sh_file.write(row)
        coefframe.loc[img,:] = all_coefs
        contframe.loc[img,:] = list(efd_cont)
    return coefframe, contframe

# transform array with fourier coefficients back to contour
def backfour(fourierarray, num_points):
    from pyefd import reconstruct_contour
    coeff_num = int(len(fourierarray) / 4)
    newarray = np.array([fourierarray[0+i:4+i] for i in range(coeff_num)])
    contour = reconstruct_contour(newarray, locus=(0, 0), num_points=num_points)
    return contour


# constants
MAIN_DIR = '/Users/Thomsn/Desktop/posterscent/figures_man'
SHAPE_FILE = 'data/shape_petals.csv'
SCALE_FILE = 'data/shape_scale.csv'


# main
mpl.rcParams['pdf.fonttype'] = 42

os.chdir(MAIN_DIR)

#### SHAPES ####
sh_fr = pd.read_csv(SHAPE_FILE, sep = ',')
for img in set(sh_fr['image']):
    df = sh_fr.loc[sh_fr['image'] == img,:]
    shape = shapeyclean(df)
    efd_coef, efd_cont = efdala(shape, 500, 40)
#     fig = efd_plot(efd_cont)


sh_cofr, sh_cnfr = coeffary(sh_fr, 40)
sh_cnfr = sh_cnfr/200
x = sh_cofr.values
y = ['a','b'] * int(len(sh_cofr.index.values) / 2)

np.set_printoptions(precision=100)

# Transformation and backtransformation do not run properly
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
dat = np.dot(x[-1] - pca.mean_, pca.components_.T)
pca_score = pca.explained_variance_ratio_
newflower = pca.inverse_transform([-0.29372032868352804, 0.15231995000245654])
newflower = backfour(newflower, 500)

nf = np.subtract(np.dot(dat, pca.components_),x[-1])
nn = np.dot(dat, pca.components_) + nf
nc = backfour(nf, num_points=500)
efd_plot(nc)

newflower = newflower.T

fig = plt.figure(figsize = (10,10))
ax = fig.add_subplot(1,1,1) 
ax.plot(newflower[0],newflower[1])
ax.grid()

# contour = reconstruct_contour(newarray, locus=(0, 0), num_points=num_points)

# plot pcas
pdf_name = f'figs/shape_pca23.pdf'
pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_name)

fig = plt.figure(figsize = (100,70))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 2', fontsize = 15)
ax.set_ylabel('Principal Component 3', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
for img in y:
    flower = principalDf.loc[img,:]
    xes = [flower[1] + xx[0] for xx in sh_cnfr.loc[img,:]]
    yes = [flower[2] + xx[1] for xx in sh_cnfr.loc[img,:]]
    ax.plot(xes, yes, linewidth=2)
ax.grid()
pdf.savefig(fig)
pdf.close()