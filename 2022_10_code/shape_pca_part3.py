#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 16:43:06 2020

this script is used to analyze and plot morphometrics of primula
for the manuscript with Dötterl, Schaefer et al.. This is part 2 and 
is thought to produce the final plots after analysis in R (momocs).

@author: Thomsn
"""
# imports
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


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
    for img in sorted(set(sh_fr['image'])):
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

# plot PCA with shapes:
def pcaplotti(shape_frame, pca_frame, pcaeig_frame):
    import matplotlib.backends.backend_pdf
    COLORS = ['darkorchid', 'gold']
    TAXA = ['hirsuta', 'lutea']
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1,1,1) 
    
    ax.axis('equal')
    tx1, tx2 = True, True
    for img in shape_frame.index:
        flower = pca_frame.loc[pca_frame['file'] == img,:]
        xes = [float(flower['PC1'] + xx[0]) for xx in shape_frame.loc[img,:]]
        yes = [float(flower['PC2'] + xx[1]) for xx in shape_frame.loc[img,:]]
        taxon = flower['species'].values[0]
        tax_bool = [i for i, tx in enumerate(TAXA) if tx == taxon][0]
        col = COLORS[tax_bool]
        ax.plot(xes, yes, linewidth=.5, color = 'black')
        if tx1:
            if taxon == 'lutea':
                ax.fill(xes, yes, color = col, alpha=.5, label = f'P. {flower["species"].values[0]}')
                tx1 = False
            else:
                ax.fill(xes, yes, color = col, alpha=.5)
        elif tx2:
            if taxon == 'hirsuta':
                ax.fill(xes, yes, color = col, alpha=.5, label = f'P. {flower["species"].values[0]}')
                tx2 = False
            else:
                ax.fill(xes, yes, color = col, alpha=.5)
        else:
            ax.fill(xes, yes, color = col, alpha=.5)
    ax.grid()
    ax.legend(loc='upper left', title='Species')
    ax.set_xlabel(f'PC1 / {round(pcaeig_frame.iat[0,1]*100, 2)} %')
    ax.set_ylabel(f'PC2 / {round(pcaeig_frame.iat[1,1]*100, 2)} %')

    pdf_name = f'figs/shape_primula_pc_1_2.pdf'
    pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_name)
    pdf.savefig(fig)
    pdf.close()
    return fig

# constants
MAIN_DIR = '/Users/Thomsn/Desktop/posterscent/figures_man'
SHAPE_FILE = 'data/shape_petals.csv'
SCALE_FILE = 'data/shape_scale.csv'
PCSHP_FILE = 'data/shape_pca_axes.csv'
PCPCT_FILE = 'data/shape_pca_power.csv'
PCDAT_FILE = 'data/shape_pca_data.csv'
OG_IMAGES = 'data/shape_fig_no.csv'


# main
mpl.rcParams['pdf.fonttype'] = 42

os.chdir(MAIN_DIR)

#### SHAPES ####
sh_fr = pd.read_csv(SHAPE_FILE, sep = ',')
nimg_fr = pd.read_csv(OG_IMAGES, sep = ',', header = None)

sh_bool = [True if sf not in nimg_fr.values else False for sf in sh_fr['image']]
sh_fr = sh_fr.loc[sh_bool,:].copy()

for img in set(sh_fr['image']):
    df = sh_fr.loc[sh_fr['image'] == img,:]
    shape = shapeyclean(df)
    efd_coef, efd_cont = efdala(shape, 500, 40)
#     fig = efd_plot(efd_cont)

sh_cofr, sh_cnfr = coeffary(sh_fr, 40)
sh_cnfr = sh_cnfr/50
x = sh_cofr.values

#### PCA ####
pd_fr = pd.read_csv(PCDAT_FILE, sep = ',')
pd_fr['file'] = sh_cnfr.index

pp_fr = pd.read_csv(PCPCT_FILE, sep = ',')

pc_fr = pd.read_csv(PCSHP_FILE, sep = ',')

pcaplotti(sh_cnfr, pd_fr, pp_fr)






