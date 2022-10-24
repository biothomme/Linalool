#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 09:16:46 2019

@author: Thomsn
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import manifold
from sklearn.metrics import euclidean_distances
from sklearn.decomposition import PCA

plant_genus = 'silene'
num = 11

os.chdir('/Users/Thomsn/Desktop/posterscent')

with open(f'{plant_genus}_plain.csv') as scentfile:
    df = pd.read_csv(scentfile, sep = ';')

if plant_genus == 'silene':
    with open(f'{plant_genus}_plain2.csv') as scentfile:
        df = pd.read_csv(scentfile, sep = ';')

    with open(f'sil_sam_dat.csv') as infofile:
        info_df = pd.read_csv(infofile, sep = ';')
        
    with open(f'sil_pop_dat.csv') as popfile:
        pop_df = pd.read_csv(popfile, sep = ';')

compounds = df.iloc[1, 1:]
specimen = df.iloc[5:, 0]
scents = df.iloc[5:, 1:].apply(pd.to_numeric)
scents = scents.fillna(0)
total_scents = scents.sum(1)
norm_scents = np.sqrt(scents.divide(total_scents, axis=0))

sp_color = specimen.str.contains('prihyb') # Primula
sk_color = specimen.str.contains('2019')
sp_color[sp_color * sk_color] = ('black')
so_color = specimen.str.contains('prihir')
sp_color[so_color == True] = ('green')
sp_color[so_color * sk_color] = ('yellowgreen')
st_color = specimen.str.contains('prilut')
sp_color[st_color == True] = ('red')
sp_color[st_color * sk_color] = ('coral')

# sp_color = specimen.str.contains('K') # SIlene
# sp_color[sp_color == True] = ('green')
# so_color = specimen.str.contains('S')
# sp_color[so_color == True] = ('black')


##### PCA #####
pca = PCA(n_components=2)
scent_pca = pca.fit(norm_scents.T)

pca_points = pd.DataFrame(scent_pca.components_)
pc_ev1, pc_ev2 = scent_pca.explained_variance_ratio_*100.00
print(scent_pca.singular_values_)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel(f'Principal Component 1 - ({pc_ev1:.2f} %)', fontsize = 15)
ax.set_ylabel(f'Principal Component 2 - ({pc_ev2:.2f} %)', fontsize = 15)
ax.set_title(f'PCA {plant_genus}', fontsize = 20)

colors = ['red', 'coral', 'green', 'yellowgreen', 'black']
targets = ['lutea18', 'lutea19', 'hirsuta18', 'hirsuta19', 'hybrid19']

# colors = ['black', 'green']
# targets = ['K', 'S']

# ax.scatter(pca_points.loc[0],\
#            pca_points.loc[1],\
#            #c = color,\
#            s = 50)
for target, color in zip(targets, colors):
    bool = [i == color for i in sp_color]
    indicesToKeep = [i for indx,i in enumerate(list(pca_points)) if bool[indx] == True]
    ax.scatter(pca_points.loc[0,indicesToKeep],
               pca_points.loc[1,indicesToKeep],
               c=color,
               s=50,
               lw=0,
               label='NMDS')
ax.legend(targets)
ax.grid()
plt.savefig(f'figs/{plant_genus}_PCA_{num}.pdf', format='pdf')


##### Prepare NMDS #####
# scents = norm_scents # optional
similarities = euclidean_distances(norm_scents)

mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9,
                   dissimilarity="precomputed", n_jobs=1)
pos = mds.fit(similarities).embedding_
nmds = manifold.MDS(n_components=2, metric=False, max_iter=3000, eps=1e-12,
                dissimilarity="precomputed", n_jobs=1,
                n_init=1)
npos = nmds.fit_transform(similarities, init=pos)

# Rescale the data
# pos *= np.sqrt((scents ** 2).sum()) / np.sqrt((pos ** 2).sum())
# npos *= np.sqrt((scents ** 2).sum()) / np.sqrt((npos ** 2).sum())
# Rotate the data
clf = PCA(n_components=2)

npos = clf.fit_transform(npos)


##### NMDS #####
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel(f'NMDS 1', fontsize = 15)
ax.set_ylabel(f'NMDS 2', fontsize = 15)
ax.set_title(f'NMDS {plant_genus}_norm', fontsize = 20)


for target, color in zip(targets, colors):
    bool = [i == color for i in sp_color]
    indicesToKeep = [i for indx,i in enumerate(range(len(npos[:,1]))) if bool[indx] == True]
    ax.scatter(npos[indicesToKeep, 0],
               npos[indicesToKeep, 1],
               c=color,
               s=50,
               lw=0,
               label='NMDS')
ax.legend(targets)
for i, spe in enumerate(specimen):
    ax.annotate(spe,
                ((npos[i, 0]+.015),
                  (npos[i, 1]+.005)),
                alpha = .5)
ax.grid(alpha = .2)
plt.savefig(f'figs/{plant_genus}_NMDS_{num}.pdf', format='pdf')



#### SILENE!!! #####



popsns = info_df['population']
keysnesses = info_df['KS']

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel(f'Principal Component 1 - ({pc_ev1:.2f} %)', fontsize = 15)
ax.set_ylabel(f'Principal Component 2 - ({pc_ev2:.2f} %)', fontsize = 15)
ax.set_title(f'PCA {plant_genus}', fontsize = 20)


colors = ['red', 'green']
targets = ['K', 'S']

# ax.scatter(pca_points.loc[0],
#             pca_points.loc[1],
#             c = 'black',
#             s = 1000)
for target, color in zip(targets, colors):
    indicesToKeep = [index for index,i in enumerate(keysnesses) if i == target]
    print(indicesToKeep)
    ax.scatter(pca_points.loc[0,indicesToKeep],
               pca_points.loc[1,indicesToKeep],
               c=color,
               s=50,
               lw=0,
               label='PCA')
ax.legend(targets)
ax.grid()
plt.savefig(f'figs/{plant_genus}_PCA_{num}.pdf', format='pdf')




# colors = ['black', 'green']
# targets = ['K', 'S']





##### NMDS #####
which_group = lambda x: pop_df.loc[:,'group'][pop_df.population == x].values
to_unk = lambda x: ['unk'] if not x else x
to_str = lambda x: str(x[0])

popgroups = info_df['population'].map(which_group)
popgroups = popgroups.map(to_unk)
popgroups = popgroups.map(to_str)
marker_list = ['^','v', 'o', '<', 's','>', '*', 'd', '.', 'h', 'D', 'P']
popuni = popgroups.unique()
markers = marker_list[:len(popuni)]

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel(f'NMDS 1', fontsize = 15)
ax.set_ylabel(f'NMDS 2', fontsize = 15)
ax.set_title(f'NMDS {plant_genus}_norm', fontsize = 20)


for target, color in zip(targets, colors):
    indicesToKeep1 = [index for index,i in enumerate(keysnesses) if i == target]
    for groppo, mark in zip(popuni, markers):
        indicesToKeep = [index for index,i in zip(popgroups[indicesToKeep1].index,
                                                  popgroups[indicesToKeep1]) if i == groppo]

        ax.scatter(npos[indicesToKeep, 0],
                   npos[indicesToKeep, 1],
                   c=color,
                   marker=mark,
                   s=50,
                   lw=0,
                   label=f'{groppo} ({target})')
handles, labels = ax.get_legend_handles_labels()
first_leg = plt.legend(handles = [x for index,x in enumerate(handles) if index % 5 == 0],
          labels = targets,
          bbox_to_anchor=(1.05, .45),
          loc='upper left',
          borderaxespad=0.,
          title = 'Substrate')
plt.gca().add_artist(first_leg)



ax.legend(handles = handles[:5],
          bbox_to_anchor=(1.05, .55),
          loc='lower left',
          borderaxespad=0.,
          labels = [x[:3] for x in labels[:5]],
          title = 'Group')
leg = ax.get_legend()
for leghandle in leg.legendHandles:
    leghandle.set_color('grey')



for i, spe in enumerate(popsns):
    ax.annotate(spe,
                ((npos[i, 0]+.015),
                  (npos[i, 1]+.005)),
                alpha = .5)
ax.grid(alpha = .2)
plt.savefig(f'figs/{plant_genus}_NMDS_{num}.pdf', format='pdf', bbox_inches='tight')



