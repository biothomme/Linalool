#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 14:23:11 2020

this script is used to plot spectrometric data from flowers
for the manuscript with Dötterl, Schaefer et al.

@author: Thomsn
"""
# imports
import os
import numpy as np
import pandas as pd
import matplotlib as mpl


# functions
# getting rid of unnecessary columns
def remove_wls(colorframe):
    colorframe.columns = [cl.rstrip() for cl in colorframe.columns]
    colorframe.index = list(colorframe['WL'])
    un_cols = [uc for uc in colorframe.columns if 'WL' in uc.upper()]

    return colorframe.drop(columns=un_cols).copy()

# subsetting data within visible range
def visira(colorframe):
    cutbool = [True if  el>=300 and el<=700 else False for el in colorframe.index]
    cutframe = colorframe.loc[cutbool,:].copy()
    return cutframe

# normalize each spectrum by subt. min and div. max
def normawave(colorframe):
    newframe = colorframe.copy()
    for wave in colorframe.columns:
        min_fq = min(colorframe[wave])
        pos_fq = (colorframe[wave] - min_fq)
        max_fq = max(pos_fq)
        newframe[wave] = pos_fq / max_fq
    return newframe

# get a summarizing header
def headmecool(colorframe):
    cfr_split = [cfc.split('_') for cfc in colorframe.columns]
    species = ['. '.join(hd[:2]) for hd in cfr_split]
    area = ['. '.join(hd[2:-1]) for hd in cfr_split]
    ind = [hd[-1] for hd in cfr_split]
    newframe = colorframe.copy()
    header = [np.array(area),
              np.array(species),
              np.array(ind)]
    header = pd.MultiIndex.from_arrays(header, names=('area', 'species', 'individual'))
    newframe.columns = header
    return newframe

# plotting a single subset - only if subset!!!
def single_plot(colorframe, p_threshold):
    import matplotlib.pyplot as plt
        
    speciess = set(colorframe.columns.get_level_values(0))
    if 'P. lutea' in list(speciess):
        COLORS = ['darkorchid', '#FCC200']
    elif 'G. acaulis' in list(speciess):
        COLORS = ['mediumblue', 'darkgreen']
    else:
        COLORS = ['darkorchid', 'coral']

    y = colorframe.index
        
    fig = plt.figure(figsize=(8, 6))
    ax = plt.subplot(111)
        
    # wavelengths with significant differences - threshold .001!
    try:
        sign_wls = wlanova(colorframe)
    except IndexError:
        sign_wls = []
    else:
        sign_wls = [yv for sw, yv in zip(sign_wls, y) if sw < p_threshold]
        
    for sign_wl in sign_wls:
        if sign_wl > 380:
            color = wavelength_to_rgb(sign_wl, gamma=0.8)
        else:
            color = 'gray'
        plt.axvline(sign_wl, c=color, alpha=.05, linewidth = 2)
        
    # lines and mean lines
    for i, species in enumerate(speciess):
        x = colorframe[species]
        inds = set(x.columns)
        for ind in inds:
            ax.plot(y, x[ind], color = COLORS[i], alpha = .2)
        ax.plot(y, x.mean(axis=1).values, color = COLORS[i], label = species, linewidth = 2)

    ax.legend(loc='upper left', title='Species')
    ax.set_xlabel('Wavelength / nm')
    ax.set_ylabel('Diffuse reflexion')
    return fig


# plotting all subsets after each other
def plotall(colorframe, p_threshold):
    import matplotlib.pyplot as plt
    import matplotlib.backends.backend_pdf
    areas = set(colorframe.columns.get_level_values(0))
    speciess = [sp.lower().replace('. ','_') for sp in set(colorframe.columns.get_level_values(1))]
    pdf_name = f'figs/color_{"|".join(speciess)}_{"|".join(areas)}_{str(p_threshold).replace("0.","")}.pdf'
    pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_name)
    for area in areas:
        fig = single_plot(colorframe[area], p_threshold)
        pdf.savefig(fig)

        # pca plot
        color_data = colorframe[area].transpose()
        min_color_data = color_data.values

        index_names = [f'{idx[0]}_{idx[1]}' for idx in color_data.index]
        assignments = pd.DataFrame(data=list(color_data.index.get_level_values(0)),
                                   columns=['population'],
                                   index=index_names)

        if len(assignments['population'].unique()) > 1:
            print(assignments['population'].unique())
            try:
                fig = pca_snsplot(color_data=min_color_data,
                                  assignments=assignments['population'],
                                  index_names=index_names)
            except np.linalg.LinAlgError:
                pass
            else:
                pdf.savefig(fig)
    pdf.close()
    return


# shapes - imported
def assign_shapes_or_col(groups_for_shapes, colors=False):
    '''
    Parameters
    ----------
    groups_for_shapes : a list of groups that the plot should be orderered by
    colors: defines, if one want to obtain colors (from 0 to 1) or shapes

    Returns
    -------
    a list of shapes (or colors) in kongruent groups

    '''
    all_markers = ['*', 'P', 'X', 'o', '+']
    if colors:
        all_markers = np.array(range(len(groups_for_shapes)))\
            / len(list(set(groups_for_shapes)))
    given_markers = {group: all_markers[i]
                     for i, group in enumerate(list(set(groups_for_shapes)))}

    return [given_markers[el] for el in groups_for_shapes]

# oneway ANOVA to find significant wavelengths, usually with bonferroni-correction
def wlanova(colorframe, bonferroni=True):
    from scipy.stats import f_oneway
    from statsmodels.sandbox.stats.multicomp import multipletests

    speciess = list(set(colorframe.columns.get_level_values(0)))
    p_list = []
    f_list = []
    for _, row in colorframe.iterrows():
        f, p = f_oneway(row[speciess[0]].values, row[speciess[1]].values)
        f_list.append(f)
        p_list.append(p)
    if bonferroni:
        p_list = multipletests(p_list, method='bonferroni')[1]
    return p_list


# convert file to proper format for the 2018 dataset
def convile(flow_file):
    import re

    GENERA = ['Silene', 'Gentiana', 'R.']
    ROCODE = {'i': 'ventr', 'a': 'dors'}
    GECODE = {'grün': 'ventmed',
              'AußenMedial': 'dorsmed',
              'SpitzeInnen': 'ventapi',
              'außenApikal': 'dorsapi'}
    colorframe = pd.read_csv(flow_file, sep=';')
    colorframe.columns = [f'WL{i}' if 'nm' in smp else smp for i, smp in enumerate(colorframe.columns)]
    colorframe.columns = [f'WL{i}' if 'Unnamed' in smp else smp for i, smp in enumerate(colorframe.columns)]
    colorframe.columns = [f'WL' if smp == 'WL0' else smp for i, smp in enumerate(colorframe.columns)]
    for genus in GENERA:
        genus_bool = [smp for smp in colorframe.columns if genus in smp]
        genus_frame = colorframe[genus_bool].copy()
        if genus == 'R.':
            areas = [ROCODE[smp.split('_')[1]] for smp in genus_frame.columns]
            species = ['_'.join(smp.split(' ')[:-1]).replace('.','') for smp in genus_frame.columns]
            inds = [smp.split('_')[0].split(' ')[-1] for smp in genus_frame.columns]
            out_filename = 'data/color_rhododendron.csv'
        elif genus == 'Silene':
            areas = ['petal' for smp in genus_frame.columns]
            species = ['_'.join(['S. a',smp.split(' ')[2]]) for smp in genus_frame.columns]
            inds = [smp.split(' ')[-1] for smp in genus_frame.columns]
            out_filename = 'data/color_silene.csv'
        else:
            areas = [GECODE[smp.split('_')[1]] for smp in genus_frame.columns]
            species = ['G_clusii' if 'clusii' in smp else 'G_acaulis' for smp in genus_frame.columns]
            inds = [re.search('[0-9]+', smp).group(0) for smp in genus_frame.columns]
            out_filename = 'data/color_gentiana.csv'
        genus_frame.columns = [f'{sp}_{ar}_{ind}' for sp, ar, ind in zip(species, areas, inds)]
        genus_frame = pd.concat([colorframe['WL'].copy(), genus_frame], axis=1)
        genus_frame.to_csv(out_filename, sep=';', index=False)
    return


# sns_pcaplot
def pca_snsplot(color_data,
                assignments,
                index_names,
                pcomp_a=1,
                pcomp_b=2,
                shapes=None):
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                  AnnotationBbox)
    import seaborn as sns
    import pandas as pd
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer


    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp = imp.fit(color_data)
    color_data = imp.transform(color_data)

    pca = PCA(svd_solver='full').fit(color_data.transpose())

    COLOR_MAP = 'spring'#'tab20'

    pc_a = f'PC{pcomp_a}'
    pc_b = f'PC{pcomp_b}'

    colors = assign_shapes_or_col(assignments, colors=True)
    print(colors)
    col_dic = {asg: co for asg, co in zip(assignments.keys(), colors)}
    cmap = cm.get_cmap(COLOR_MAP, 256)

    cmap_dic = {asg: cmap(co) for asg, co in zip(assignments.values, colors)}
    color_order = [cl for i, cl in enumerate(colors) if cl not in colors[:i]]

    pca_values = pca.transform(color_data.transpose())
    pca_plotting_data = pd.DataFrame(zip(pca_values[:, pcomp_a-1],
                                         pca_values[:, pcomp_b-1],
                                         assignments.values,
                                         index_names,
                                         index_names),
                                     columns=[pc_a,
                                              pc_b,
                                              'colors',
                                              'names',
                                              'place_key'],
                                     index=index_names)

    fig = plt.figure(figsize=(8, 6))

    ax = plt.subplot(1, 1, 1)

    if isinstance(shapes, type(None)):
        shapes = ['sample'] * len(list(assignments.values))
        new_shapes = ['^'] * len(list(assignments.values))
    else:
        new_shapes = assign_shapes_or_col(shapes)

    # plot the clouds
    if 'G. acaulis' not in list(assignments.values):
        hybrid_free = [nim for nim in index_names if 'x' not in str(nim)]
        pca_hybrid_free = pca_plotting_data.loc[hybrid_free, :]
        ax = sns.kdeplot(data=pca_hybrid_free,
                         x=pc_a,
                         y=pc_b,
                         hue='colors',
                         palette=cmap_dic,
                         ax=ax,
                         fill=True,
                         levels=2,
                         thresh=.1,
                         alpha=.2)

    # plot the points
    s = 40

    for i, (_, data_row) in enumerate(pca_plotting_data.iterrows()):
        ax.scatter(x=data_row[pc_a],
                   y=data_row[pc_b],
                   c=colors[i],
                   cmap=COLOR_MAP,
                   vmin=0,
                   vmax=1,
                   alpha=.9,
                   s=s,
                   marker=new_shapes[i],
                   edgecolor='#333333',
                   lw=.3)

    # axes primula:
    if 'Ph' in assignments.values:
        ax.set_xlim(-9, 18)
        ax.set_ylim(-8, 12)

    # axes rhodo:
    if 'Rh' in assignments.values:
        ax.set_xlim(-16, 18)
        ax.set_ylim(-15, 18)

    ax.axvline(0, alpha=.2, lw=1, c='grey')
    ax.axhline(0, alpha=.2, lw=1, c='grey')

    # plot the legend
    handles = list()
    for ass, col in sorted(set(zip(assignments, colors))):
        new_handle = ax.plot([], [], lw=2, c=cmap(col), label=ass)
        handles.append(new_handle[0])

    legend1 = ax.legend(handles=handles,
                        bbox_to_anchor=(1.02, .99),
                        loc='upper left',
                        borderaxespad=0.,
                        title="Species")
    ax.add_artist(legend1)

    handles = list()
    for shass, shape in sorted(set(zip(shapes, new_shapes))):
        new_handle = ax.plot([],
                             [],
                             ls='',
                             c='black',
                             marker=shape,
                             markersize=10,
                             label=shass)
        handles.append(new_handle[0])


    ax.legend(handles=handles,
              bbox_to_anchor=(1.02, .01),
              loc='lower left',
              borderaxespad=0.,
              title="Plot")

    return fig


# color convert to wavelength
def wavelength_to_rgb(wavelength, gamma=0.8):
    '''This converts a given wavelength of light to an 
    approximate RGB color value. The wavelength must be given
    in nanometers in the range from 380 nm through 750 nm
    (789 THz through 400 THz).

    Based on code by Dan Bruton
    http://www.physics.sfasu.edu/astro/color/spectra.html
    '''
    wavelength = float(wavelength)
    if wavelength >= 380 and wavelength <= 440:
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
        G = 0.0
        B = (1.0 * attenuation) ** gamma
    elif wavelength >= 440 and wavelength <= 490:
        R = 0.0
        G = ((wavelength - 440) / (490 - 440)) ** gamma
        B = 1.0
    elif wavelength >= 490 and wavelength <= 510:
        R = 0.0
        G = 1.0
        B = (-(wavelength - 510) / (510 - 490)) ** gamma
    elif wavelength >= 510 and wavelength <= 580:
        R = ((wavelength - 510) / (580 - 510)) ** gamma
        G = 1.0
        B = 0.0
    elif wavelength >= 580 and wavelength <= 645:
        R = 1.0
        G = (-(wavelength - 645) / (645 - 580)) ** gamma
        B = 0.0
    elif wavelength >= 645 and wavelength <= 750:
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        R = (1.0 * attenuation) ** gamma
        G = 0.0
        B = 0.0
    else:
        R = 0.0
        G = 0.0
        B = 0.0
    return (R, G, B)


# constants
MAIN_DIR = '/Users/Thomsn/Desktop/posterscent/figures_man'
PRIM_FILE = 'data/color_primula.csv'
MIX_FILE = 'data/color_2018.csv'
SIL_FILE = 'data/color_silene.csv'
GEN_FILE = 'data/color_gentiana.csv'  # 2018
GEN2_FILE = 'data/color_gentiana2.csv'  # 2019
RHO_FILE = 'data/color_rhododendron.csv'


# main
mpl.rcParams['pdf.fonttype'] = 42

os.chdir(MAIN_DIR)

#### MIX 2018 - in a different format conversion neccessary ####
convile(MIX_FILE)

#### PRIMULA ####
fl_fr = pd.read_csv(PRIM_FILE, sep = ';')
fl_fr = remove_wls(fl_fr)
fl_fr = visira(fl_fr)
fl_fr = normawave(fl_fr)
fl_fr = headmecool(fl_fr)

plotall(fl_fr, .05)

#### GENTIANA ####
fl_fr = pd.read_csv(GEN_FILE, sep = ';')
fl_fr = remove_wls(fl_fr)
fl_fr = visira(fl_fr)

fl_fr2 = pd.read_csv(GEN2_FILE, sep = ';')
fl_fr2 = remove_wls(fl_fr2)
fl_fr2 = visira(fl_fr2)

# Gentiana has samples 2018 and 2019
fl_fr = pd.concat([fl_fr.copy(), fl_fr2.copy()], axis=1) 
fl_fr = normawave(fl_fr)
fl_fr = headmecool(fl_fr)

plotall(fl_fr, .05)

#### SILENE ####
fl_fr = pd.read_csv(SIL_FILE, sep = ';')
fl_fr = remove_wls(fl_fr)
fl_fr = visira(fl_fr)
fl_fr = normawave(fl_fr)
fl_fr = headmecool(fl_fr)

plotall(fl_fr, .05)

#### RHODODENDRON ####
fl_fr = pd.read_csv(RHO_FILE, sep = ';')
fl_fr = remove_wls(fl_fr)
fl_fr = visira(fl_fr)
fl_fr = normawave(fl_fr)
fl_fr = headmecool(fl_fr)

plotall(fl_fr, .05)
