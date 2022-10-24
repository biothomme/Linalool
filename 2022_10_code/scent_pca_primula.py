#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 08:56:47 2020

@author: Thomsn
"""
# github
# pip install git+https://github.com/erheron/scipy.git

# imports:
import numpy as np
import scent_genetics_primula as sgp
import matplotlib as mpl


# functions:
def load_arguments():
    '''
    this is the argparse function

    it loads the inputfile (csv) of the scents, and optional metafile, nodes
    and edges of the 
    -------
    great smell!
    '''
    import argparse as ap

    parser = ap.ArgumentParser(description=__doc__,
                               formatter_class=ap.RawDescriptionHelpFormatter)
    parser.add_argument("-i",
                        action="store",
                        dest="csv_file",
                        required=True,
                        help="Input a csv_file with scent data.")
    parser.add_argument("-m",
                        action="store",
                        dest="meta_file",
                        required=False,
                        help="Optional meta_data.")
    parser.add_argument("-g",
                        action="store",
                        dest="gene_file",
                        required=False,
                        help="Input a csv_file with the sequence summary.")

    args = parser.parse_args()

    return args.csv_file, args.meta_file, args.gene_file


def load_matrix(csv_file, infer_header=False, sep=';', first_col=False):
    '''
    Parameters
    ----------
    csv_file : containing the minimal scent file/data

    Returns a pd.DataFrame of it
    -------
    '''
    import pandas as pd

    if infer_header:
        scent_data = pd.read_csv(csv_file, sep=sep)
    else:
        scent_data = pd.read_csv(csv_file, sep=sep, header=None)

    if not first_col:
        scent_data.index = scent_data.iloc[:, 0].values
        scent_data = scent_data.iloc[:, 1:].copy()

    # scent_data.fillna(value=None, inplace=True)

    return scent_data


def semiquant(scent_data):
    '''
    divide by total voc content
    Parameters
    ----------
    scent_data : as always

    Returns
    -------
    semiquant_data : semiquantitative concentration values
    '''
    scent_data = scent_data.astype(float)
    total_bouquet = scent_data.sum(axis=1)
    semiquant_data = scent_data.div(total_bouquet, axis=0).copy()

    return semiquant_data


def log_the_data(scent_data):
    '''
    not log(!) double sqrt (!) tranform the scent data
    Parameters
    ----------
    scent_data : as always

    Returns
    -------
    newscent_data : logtransformed values
    '''
    # scent_data.replace(0, 0.00000001, inplace=True)
    newscent_data = np.sqrt(np.sqrt(scent_data.copy().astype(float)))
    newscent_data.replace(-np.inf, 0, inplace=True)

    return newscent_data


def core_matrix(scent_data):
    '''
    get a subset of the datamatrix, only involving the specimen and their
    compund concentrations

    Parameters
    ----------
    scent_data : the whole scent data matrix

    Returns
    -------
    core_matrix : matrix only involving all specimen's data
    comp_names : names of the compounds

    '''
    core_matrix = scent_data.iloc[9:, :].copy()
    comp_names = scent_data.loc['pc_name', :]

    return core_matrix, comp_names


def stdize_data(scent_data):
    '''
    to convert the scent matrix into stdized vales
    Parameters
    ----------
    scent_data : core scent data matrix (core_matrix()-output)

    Returns
    -------
    std_scent_data : similar as scent_data, but with normalized/stdized values

    '''
    import pandas as pd
    from sklearn import preprocessing

    scaler = preprocessing.StandardScaler()

    scaled_df = scaler.fit_transform(scent_data)
    std_scent_data = pd.DataFrame(scaled_df,
                                  index=scent_data.index,
                                  columns=scent_data.columns).copy()

    return std_scent_data


def pca_varplot(std_scent_data, pca):
    import matplotlib.pyplot as plt


    variances = np.std(pca.transform(std_scent_data), axis=0)**2
    components = np.arange(len(variances)) + 1

    fig = plt.figure(figsize=(8, 6))
    ax = plt.subplot(1, 1, 1)
    ax.plot(components, variances, "o-")

    ax.set_xticks(components)
    ax.set_xticklabels([str(comp) for comp in components], rotation=60)
    ax.set_xlabel('principal component')
    ax.set_ylabel('variance')

    fig.show()

    return fig


def pca_directiplot(std_scent_data,
                    pca,
                    pcomp_a,
                    pcomp_b,
                    comp_names,
                    mean_conc,
                    ax):
    import matplotlib.pyplot as plt

    pca_a_val = pca.components_[pcomp_a]
    pca_b_val = pca.components_[pcomp_b]

    xax = min([abs(x) for x in ax.get_xlim()])
    yax = min([abs(x) for x in ax.get_ylim()])
    min_ax = min([xax, yax])
    max_comp = max(list(pca_a_val)+list(pca_b_val))
    stretch_factor = min_ax / (3*max_comp)
    concis = np.log2(mean_conc)
    concis = np.log(mean_conc)

    new_concis = concis - min(concis) - np.log2(.9)
    font_sizes = np.ceil(new_concis)

    pca_a_val = [x * stretch_factor for x in pca_a_val]
    pca_b_val = [x * stretch_factor for x in pca_b_val]

    for a, b, compi, fs in zip(pca_a_val, pca_b_val, comp_names, font_sizes):
        ax.annotate(xy=(0, 0),
                    xytext=(a, b),
                    text='',
                    size=int(fs),
                    arrowprops={'arrowstyle': ']-',
                                'alpha': .5,
                                'lw': .5,
                                'color': 'lightgrey'})
    for a, b, compi, fs in zip(pca_a_val, pca_b_val, comp_names, font_sizes):
        hyp = np.sqrt(a**2+b**2)
        rotation = np.arccos(abs(a) / hyp) / np.pi * 180
        va = 'center'
        if a <= 0:
            ha = 'right'
            if b >= 0:
                rotation = 360 - rotation
        else:
            ha = 'left'
            if b <= 0:
                rotation = 360 - rotation

        tr_rotation = plt.gca().transData.transform_angles(np.array((rotation,
                                                                     )),
                                                           np.array((a, b)).reshape((1, 2)))[0]
        ax.annotate(xy=(0, 0),
                    xytext=(a, b),
                    text=compi,
                    ha=ha,
                    va=va,
                    rotation=tr_rotation,
                    rotation_mode='anchor',
                    size=int(fs+1),
                    c='grey',
                    alpha=.6)

    ax.set_xlabel(f'PC{pcomp_a}')
    ax.set_ylabel(f'PC{pcomp_b}')

    return ax


def sort_mixed_list(list_int_str):
    '''
    sorts a list that contains str and int
    Parameters
    ----------
    list_int_str :list with strings and ints

    Returns sorted list

    '''
    str_list = [str(el) for el in list_int_str]
    order = [i for el, i in
             sorted(zip(str_list, range(len(str_list))))]
    ordered_list = [list_int_str[od] for od in order]

    return ordered_list


# obtain mean_concentrations
def mean_concentrations(scent_data):
    return scent_data.mean(axis=0).copy()


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


def pca_scatplot(std_scent_data,
                 pca,
                 assignments,
                 pcomp_a,
                 pcomp_b,
                 shapes=None,
                 meta_data=None):
    import matplotlib.pyplot as plt
    from matplotlib import cm
    import pandas as pd

    COLOR_MAP = 'spring'#'tab20'

    pc_a = f'PC{pcomp_a}'
    pc_b = f'PC{pcomp_b}'

    pca_values = pca.transform(std_scent_data)
    pca_plotting_data = pd.DataFrame(zip(pca_values[:, pcomp_a-1],
                                         pca_values[:, pcomp_b-1]),
                                     columns=[pc_a, pc_b])


    colors = assign_shapes_or_col(assignments, colors=True)

    col_dic = {asg: co for asg, co in zip(assignments.keys(), colors)}


    if isinstance(shapes, type(None)):
        shapes = ['floral scent'] * len(list(assignments.values))
        new_shapes = ['^'] * len(list(assignments.values))
    else:
        new_shapes = assign_shapes_or_col(shapes)

    cmap = cm.get_cmap(COLOR_MAP, 256)
    fig = plt.figure(figsize=(8, 6))

    ax = plt.subplot(1, 1, 1)

    for i, data_row in pca_plotting_data.iterrows():
        ax.scatter(x=data_row[pc_a],
                   y=data_row[pc_b],
                   c=colors[i],
                   cmap=COLOR_MAP,
                   vmin=0,
                   vmax=1,
                   alpha=.7,
                   s=50,
                   marker=new_shapes[i])

    handles = list()
    for ass, col in sorted(set(zip(assignments, colors))):
        new_handle = ax.plot([], [], lw=2, c=cmap(col), label=ass)
        handles.append(new_handle[0])

    legend1 = ax.legend(handles=handles, bbox_to_anchor=(1.02, .99), loc='upper left', borderaxespad=0., title="Population")
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
    ax.legend(handles=handles, bbox_to_anchor=(1.02, .01), loc='lower left', borderaxespad=0., title="Soil")

    return fig


def pca_snsplot(std_scent_data,
                pca,
                assignments,
                pcomp_a,
                pcomp_b,
                shapes=None,
                meta_data=None,
                gene_file=None,
                directions=False,
                comp_names=None,
                mean_conc=None):
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                  AnnotationBbox)
    import seaborn as sns
    import pandas as pd

    COLOR_MAP = 'spring'#'tab20'

    pc_a = f'PC{pcomp_a}'
    pc_b = f'PC{pcomp_b}'

    colors = assign_shapes_or_col(assignments, colors=True)
    print(colors, type(assignments))

    col_dic = {asg: co for asg, co in zip(assignments.keys(), colors)}
    cmap = cm.get_cmap(COLOR_MAP, 256)

    cmap_dic = {asg: cmap(co) for asg, co in zip(assignments.values, colors)}
    color_order = [cl for i, cl in enumerate(colors) if cl not in colors[:i]]

    pca_values = pca.transform(std_scent_data)
    pca_plotting_data = pd.DataFrame(zip(pca_values[:, pcomp_a-1],
                                         pca_values[:, pcomp_b-1],
                                         assignments.values,
                                         meta_data['florscent_key'],
                                         meta_data['place_key']),
                                     columns=[pc_a,
                                              pc_b,
                                              'colors',
                                              'names',
                                              'place_key'],
                                     index=meta_data.index)

    fig = plt.figure(figsize=(8, 6))
    if directions == 'solely':
        fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot(1, 1, 1)

    if isinstance(shapes, type(None)):
        shapes = ['floral scent'] * len(list(assignments.values))
        new_shapes = ['^'] * len(list(assignments.values))
    else:
        new_shapes = assign_shapes_or_col(shapes)

    # plot the clouds
    hybrid_free = [nim for nim in meta_data.index if 'x' not in str(nim)]
    pca_hybrid_free = pca_plotting_data.loc[hybrid_free, :]
    # pca_hybrid_free["colors"] = pca_hybrid_free['colors'].astype('category').cat.codes
    print(cmap_dic)
    if directions != 'solely':
        ax = sns.kdeplot(data=pca_hybrid_free,
                         x=pc_a,
                         y=pc_b,
                         hue="colors",
                         palette=cmap_dic,
                         ax=ax,
                         fill=True,
                         levels=2,
                         thresh=.1,
                         alpha=.2)
    else:
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
    # add voc component directions:
    if directions:
        ax = pca_directiplot(std_scent_data,
                             pca,
                             pcomp_a,
                             pcomp_b,
                             comp_names,
                             mean_conc,
                             ax)

    # plot the points
    if directions != 'solely':
        if 'Ph' in assignments.values:
            s=25
        else:
            s=40

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

    # add pies:
    if not isinstance(gene_file, type(None)):
        gene_ind = meta_data.loc[meta_data['nucleotide_key'].dropna().index, :]
        gene_dict = {row['florscent_key']: str(int(row['nucleotide_key']))
                     for _, row in gene_ind.iterrows()}
        for indiv in gene_dict.keys():
            sel_co = pca_plotting_data.loc[pca_plotting_data['names'] == indiv,
                                           (pc_a, pc_b)]
            pie_coords = trafo_ax_fig(ax, [x for x in sel_co.values[0]])
            sample = gene_dict[indiv]

            fig = sgp.pie_the_indiv(gene_file, sample, fig, pie_coords)

        pie_coords = trafo_ax_fig(ax, [13.5, -4])
        fig = sgp.pie_the_indiv(gene_file, '4', fig, pie_coords, biggie=True)

        # plot circles around portjoch samples:
        port_bool = [indx for indx, row in pca_plotting_data.iterrows()
                     if 'Por_0' in row['place_key']]
        port_data = pca_plotting_data.loc[port_bool, :]
        for i, (_, data_row) in enumerate(port_data.iterrows()):
            if data_row.names not in list(gene_ind.index):
                ax.scatter(x=data_row[pc_a],
                           y=data_row[pc_b],
                           alpha=.9,
                           s=125,
                           marker='o',
                           edgecolors='k',
                           facecolors='none')

    ax.axvline(0, alpha=.2, lw=1, c='grey')
    ax.axhline(0, alpha=.2, lw=1, c='grey')

    # plot the legend
    if directions != 'solely':
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


# a way to transform axis coordinates to figure coordinates
def trafo_ax_fig(axis, ax_coords):
    # ax_coords = [ax_coords[1], ax_coords[0]]
    bbox = axis.get_position()
    [x0, y0], [x1, y1] = bbox.get_points()
    xa0, xa1 = axis.get_xlim()
    ya0, ya1 = axis.get_ylim()
    x_dif = (xa1-xa0) / (x1-x0)
    y_dif = (ya1-ya0) / (y1-y0)

    fig_coords = [fd + (co-cd)/ck for fd, co, cd, ck in zip([x0, y0],
                                                            ax_coords,
                                                            [xa0, ya0],
                                                            [x_dif, y_dif])]

    return [fig_coords[0], fig_coords[1]]


# subset the meta_data file and add populations line
def obtain_groups(scent_data, meta_data):
    specimen = list(scent_data.index)
    meta_bool = [i for i, row in meta_data.iterrows()
                 if row['florscent_key'] in specimen]
    meta_selection = meta_data.loc[meta_bool, :]
    population = [plo[1:3] for plo in meta_selection.index]

    meta_selection['population'] = population

    return meta_selection


def save_figures_as_pdf(function_list, fig_name):
    import matplotlib.backends.backend_pdf

    pdf = matplotlib.backends.backend_pdf.PdfPages(fig_name)
    for function in function_list:
        pdf.savefig(function, bbox_inches='tight')
    pdf.close()

    return


# change order of individuals, so that data is sorted in same way!
def sort_by_same_order(scent_data, meta_data):

    scent_data = scent_data.sort_index().copy()
    meta_data = meta_data.sort_values(by=['florscent_key']).copy()

    if list(scent_data.index) == list(meta_data['florscent_key'].values):
        return scent_data, meta_data


# make a table for summary stats
def summary_stats(csv_file, scent_data, meta_data):
    import pandas as pd

    new_file = csv_file.replace('.csv', '_summary.csv')
    minscent_data, _ = core_matrix(scent_data)
    minscent_data = semiquant(minscent_data)

    species = list(meta_data['population'].unique())
    scent_data.fillna('-', inplace=True)
    scent_data.replace('unk', 'unknown', inplace=True)

    rows = pd.MultiIndex.from_tuples([(scent_data.loc['pubchemid', i],
                                       scent_data.loc['pc_name', i],
                                       scent_data.loc['retindex', i],
                                       scent_data.loc['mol', i])
                                      for i in scent_data.columns])
    cols = [[(sp, 'mean'), (sp, 'standard_error'), (sp, 'samples/georeg')]
            for sp in species]
    cols = pd.MultiIndex.from_tuples([tp for tl in cols for tp in tl])
    summary_statistics = pd.DataFrame(columns=cols,
                                      index=rows)
    for sp in species:
        sel_meta = meta_data.loc[meta_data['population'] == sp, :]
        sel_data = minscent_data.loc[sel_meta['florscent_key'].values, :]
        for i in range(sel_data.shape[1]):
            currentindex = summary_statistics.index[i]

            nonzeros = [val for val in sel_data.iloc[:, i] if val > 0]
            nonzinx = [sel_data.index[i] for i, val in
                       enumerate(sel_data.iloc[:, i]) if val > 0]

            samps = len(nonzeros)
            if samps > 0:
                mean = np.mean(nonzeros)
                if samps > 1:
                    sdev = sum([(x - mean)**2 for x in nonzeros]) / (samps-1)
                else:
                    sdev = 0

                standard_error = sdev / samps

                geo_dat = sel_meta.loc[sel_meta['florscent_key'].isin(nonzinx), :]
                geo_dat = [pl[:3] for pl in geo_dat['place_key']]
                geo_sum = len(list(set(geo_dat)))
                
                mean = f'{mean:.5f}'
                standard_error = f'{standard_error:.10f}'
            else:
                mean = '-'
                standard_error = '-'
                samps = '-'
                geo_sum = '-'

            summary_statistics.loc[pd.IndexSlice[currentindex],
                                   pd.IndexSlice[(sp, 'mean')]] = mean
            summary_statistics.loc[pd.IndexSlice[currentindex],
                                   pd.IndexSlice[(sp, 'standard_error')]] = standard_error
            summary_statistics.loc[pd.IndexSlice[currentindex],
                                   pd.IndexSlice[(sp, 'samples/georeg')]] = f'{samps} / {geo_sum}'

    summary_statistics.to_csv(new_file)
    print(f'Summary of scent data was saved in {new_file}...')

    return summary_statistics


# make a bray curtis distance matrix
def british_columbia(scent_data):
    import itertools
    import pandas as pd
    from scipy.spatial import distance
    scent_data = scent_data.fillna(0).copy()

    bray_mat = pd.DataFrame(columns=scent_data.index, index=scent_data.index)
    for sam1, sam2 in itertools.combinations(scent_data.index, 2):
        u = scent_data.loc[sam1, :]
        v = scent_data.loc[sam2, :]
        bcd = distance.braycurtis(u, v)

        bray_mat.loc[sam1, sam2] = bcd
        bray_mat.loc[sam2, sam1] = bcd
    bray_mat = bray_mat.fillna(1)
    bray_mat = bray_mat.subtract(1).multiply(-1)

    return bray_mat


def plt_bray_curtis(dist_matrix, meta_data):
    import matplotlib.pyplot as plt
    import scipy.cluster.hierarchy as sch
    COLDIC = {1:'darkorchid', 2:'grey', 0:'gold'}
    
    def fix_verts(ax, orient=1):
        for coll in ax.collections:
            for pth in coll.get_paths():
                vert = pth.vertices
                vert[1:3,orient] = np.average(vert[1:3,orient]) 

    # Generate random features and distance matrix.
    fig = plt.figure(figsize=(8, 6))
    coldic = {spm: meta_data.loc[meta_data['florscent_key'] == spm, 'population'].values[0]
              for spm in dist_matrix.index}

    # Compute and plot the dendrogram.
    # ax2 = fig.add_axes([0.3,0.71,0.6,0.2])
    dendrog_frame = sch.linkage(dist_matrix, method='ward')

    dendrogram = sch.dendrogram(dendrog_frame)
    leaves = dendrogram['leaves']
    leaves = [dist_matrix.columns[i] for i in leaves]
    colors = [coldic[lv] for lv in leaves]
    microlors = {cl: i for i, cl in enumerate(list(set(colors)))}
    # coldic = {i: microlors[cl] for i, cl in enumerate(colors)}
    coldic = {cl: microlors[lv] for cl, lv in zip(leaves, colors)}

    dendrogram = sch.dendrogram(dendrog_frame,
                                labels=leaves,
                                orientation='left')
    ax = plt.gca()
    xlbls = ax.get_ymajorticklabels()
    for y in xlbls:
        y.set_color(COLDIC[coldic[y.get_text()]])

    return fig


# plot the whole pcas
def pca_run(std_scent_data,
            csv_file,
            comp_names,
            assignment,
            pcomp_a=1,
            pcomp_b=2,
            shapes=None,
            meta_data=None,
            gene_file=None,
            mean_conc=None,
            dist_matrix=None):
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp = imp.fit(std_scent_data)
    std_scent_data = imp.transform(std_scent_data)

    pca = PCA(svd_solver='full').fit(std_scent_data)

    # # comp_fig = csv_file.replace('.csv', '_pca_comp.pdf')
    # # save_figures_as_pdf(pca_directiplot(std_scent_data,
    # #                                     pca,
    # #                                     comp_names,
    # #                                     mean_conc),
    # #                     comp_fig)

    # var_fig = csv_file.replace('.csv', '_pca_var.pdf')
    # save_figures_as_pdf(pca_varplot(std_scent_data, pca), var_fig)

    scat_fig = csv_file.replace('.csv', '_pca.pdf')

    save_figures_as_pdf([pca_varplot(std_scent_data, pca),
                         pca_snsplot(std_scent_data,
                                     pca,
                                     assignment,
                                     pcomp_a=pcomp_a,
                                     pcomp_b=pcomp_b,
                                     shapes=shapes,
                                     meta_data=meta_data,
                                     gene_file=gene_file),
                         pca_snsplot(std_scent_data,
                                     pca,
                                     assignment,
                                     pcomp_a=pcomp_a,
                                     pcomp_b=pcomp_b,
                                     shapes=shapes,
                                     meta_data=meta_data,
                                     gene_file=gene_file,
                                     directions=True,
                                     comp_names=comp_names,
                                     mean_conc=mean_conc),
                         plt_bray_curtis(dist_matrix, meta_data),
                         pca_snsplot(std_scent_data,
                                     pca,
                                     assignment,
                                     pcomp_a=pcomp_a,
                                     pcomp_b=pcomp_b,
                                     shapes=shapes,
                                     meta_data=meta_data,
                                     directions='solely',
                                     comp_names=comp_names,
                                     mean_conc=mean_conc)],
                         scat_fig)

    return




# constants:


# main:
mpl.rcParams['pdf.fonttype'] = 42

csv_file, meta_file, gene_file = load_arguments()
old_scent_data = load_matrix(csv_file)

scent_data, comp_names = core_matrix(old_scent_data.copy())
scent_data = semiquant(scent_data)
distance_matix = british_columbia(scent_data)

scent_data = log_the_data(scent_data)

mean_conc = mean_concentrations(scent_data)

meta_data = load_matrix(meta_file, infer_header=True)
meta_data = obtain_groups(scent_data, meta_data)

# summary_statistics = summary_stats(csv_file, old_scent_data, meta_data)

scent_data, meta_data = sort_by_same_order(scent_data, meta_data)

std_scent_data = stdize_data(scent_data)

pca_run(std_scent_data,
        csv_file,
        comp_names,
        meta_data['population'],
        pcomp_a=1,
        pcomp_b=2,
        shapes=meta_data['bed_rock'],
        meta_data=meta_data,
        gene_file=gene_file,
        mean_conc=mean_conc,
        dist_matrix=distance_matix)
