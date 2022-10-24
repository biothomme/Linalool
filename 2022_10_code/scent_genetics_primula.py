#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 14:04:21 2020

scents_genetics_primula includes important functions to plot the genetic
data pies for the floral scent pca (primula).

@author: Thomsn
"""
# imports:
import numpy as np
from Bio import AlignIO


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
                        help="Input a csv_file with the sequence_summary.")

    args = parser.parse_args()

    return args.csv_file


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


def aligns(meta_file, meta_data):
    fasta_base = '/'.join(meta_file.split('/')[:-1])
    fastas = dict()
    for _, meta_row in meta_data.iterrows():
        f_file = meta_row['file']
        f_name = meta_row.name
        fasta_file = f'{fasta_base}/{f_file}'
        fastas[f_name] = AlignIO.read(fasta_file, 'fasta')

    return fastas


def get_polymophic_sites(alignment):
    snps = dict()
    for i in range(alignment.get_alignment_length()):
        locus = alignment[:, i:i+1]
        if len(set(list(locus[:, 0]))) != 1:
            snps[i] = locus
    return snps


# build a dict to stdize the sequence headers between all alignments
def stdize_seq_names(meta_data):
    stds = dict()
    for i, row in meta_data.iterrows():
        row = row['1':'p_lutea_ref']
        for ind in row.index:
            stds[row[ind]] = ind
    return stds


# build a standard for the species lutea/hirsuta
def thats_the_real_stan(snps_dict):
    AMBIGUITIES = {'at': 'w',
                   'ct': 'y',
                   'ag': 'r',
                   'gt': 'k',
                   '-c': 'c',
                   'cc': 'c',
                   'gg': 'g',
                   'aa': 'a'}

    stdizer = dict()
    for gene, gdict in snps_dict.items():
        genstd = dict()
        if 'p_lutea_ref' in gdict.keys():
            lu = [b if b in 'atgc-' else bb for b, bb in
                  zip(gdict['p_lutea_ref'], gdict['2'])]
        else:
            lu = gdict['2']
        if 'p_hirsuta_ref' in gdict.keys():
            hi = [b if b in 'atgc-' else bb for b, bb in
                  zip(gdict['p_hirsuta_ref'], gdict['1'])]
        else:
            hi = gdict['1']
        for i in range(len(lu)):
            locstd = dict()
            variants = set([v[i] for v in gdict.values()])
            for var in variants:
                if (lu[i] == var and hi[i] == var):
                    locstd[var] = 'ancestral'
                elif lu[i] == var:
                    locstd[var] = 'lutea'
                elif hi[i] == var:
                    locstd[var] = 'hirsuta'
                elif (gdict['2'][i] == var and var in 'atgc-'):
                    locstd[var] = 'lutea_local'
                elif (gdict['1'][i] == var and var in 'atgc-'):
                    locstd[var] = 'hirsuta_local'
                elif AMBIGUITIES[''.join(sorted([lu[i], hi[i]]))] == var:
                    locstd[var] = 'ambiguity'
                else:
                    locstd[var] = 'unspecified'
            genstd[i] = locstd
        stdizer[gene] = genstd

    return stdizer


# return a dict with a dict per alignment storing the data for snps
def all_poymorphisms(alignments, std_dict=None):
    import pandas as pd
    snps_dict = dict()
    snp_names = dict()
    for names, alignment in alignments.items():
        snp_data = get_polymophic_sites(alignment)
        all_snps = list(snp_data.keys())
        snp_dict = dict()
        for sequence in alignment:
            snp_dict[std_dict[sequence.name]] = [b for i, b in
                                                 enumerate(sequence.seq)
                                                 if i in all_snps]
        snps_dict[names] = snp_dict
        snp_names[names] = list(snp_data.keys())
    pd.DataFrame(snps_dict).to_csv("snps.txt")
    print(pd.DataFrame(snps_dict), snp_names)
    return snps_dict, snp_names


# add hatch to pie:
def hatchcake(piechart, hatch_list, hatch_color):
    for i, htch in enumerate(hatch_list):
        if htch != 'none':
            piechart[0][i].set_hatch(htch)
            piechart[0][i].set_edgecolor(hatch_color[i])

    return piechart


# plot a pie
def coffeetime(ax,
               fractions,
               size,
               colors,
               width,
               lw,
               alpha,
               labels=None,
               labeldistance=None,
               textprops=None,
               startangle=0):

    piechart = ax.pie(fractions,
                      radius=size,
                      colors=colors,
                      counterclock=False,
                      wedgeprops=dict(width=width,
                                      edgecolor='k',
                                      linewidth=lw,
                                      alpha=alpha),
                      labels=labels,
                      labeldistance=labeldistance,
                      textprops=textprops,
                      startangle=startangle)

    return ax, piechart


# plot the pies:
def pie_time(alignments, meta_data, ind, figure, coords, biggie=False):
    COLORDIC = {'ancestral': 'palegreen',
                'lutea': 'gold',
                'hirsuta': 'darkorchid',
                'unspecified': 'grey',
                'hirsuta_local': 'darkorchid',
                'lutea_local': 'gold',
                'ambiguity': 'gold'}
    HATCHDIC = {'ancestral': 'none',
                'lutea': 'none',
                'hirsuta': 'none',
                'unspecified': 'none',
                'hirsuta_local': 'white',
                'lutea_local': 'white',
                'ambiguity': 'darkorchid'}
    ALPHADIC = {'ancestral': 'none',
                'lutea': 'none',
                'hirsuta': 'none',
                'unspecified': 'none',
                'hirsuta_local': '///////',
                'lutea_local': '///////',
                'ambiguity': '///////'}

    SIZE = .45
    INSET_SIZE = .06
    ALPH = .6
    if biggie:
        INSET_SIZE = .25

    std_dict = stdize_seq_names(meta_data)
    snps_dict, snp_names = all_poymorphisms(alignments, std_dict=std_dict)
    snp_names = {nam: [el+meta_data.loc[nam, 'std_start'] for el in lst]
                 for nam, lst in snp_names.items()}

    standardizing_dict = thats_the_real_stan(snps_dict)

    chplast_genes = meta_data.index[meta_data['organelle'] == 'chloroplast']
    nucl_genes = meta_data.index[meta_data['organelle'] != 'chloroplast']

    cpsnps = [len(standardizing_dict[cpg]) for cpg in chplast_genes]
    cpfracs = [1] * sum(cpsnps)

    ncsnps = [len(standardizing_dict[ncg]) for ncg in nucl_genes]
    ncfracs = [1] * sum(ncsnps)

    fig = figure

    ax_shape = [co - INSET_SIZE/2 for co in coords] + [INSET_SIZE, INSET_SIZE]
    ax = fig.add_axes(ax_shape)

    cpolors = []
    calleles = []
    chatch = []
    calpha = []

    ncolors = []
    nalleles = []
    nhatch = []
    nalpha = []

    for cpg in chplast_genes:
        if biggie:
            calleles += snp_names[cpg]
        snp_list = snps_dict[cpg][ind]
        stdizer = standardizing_dict[cpg]
        for i, snp in enumerate(snp_list):
            cpolors.append(COLORDIC[stdizer[i][snp]])
            chatch.append(HATCHDIC[stdizer[i][snp]])
            calpha.append(ALPHADIC[stdizer[i][snp]])
        #     calleles.append(snp.upper())

    for ncg in nucl_genes:
        if biggie:
            nalleles += snp_names[ncg]
        snp_list = snps_dict[ncg][ind]
        stdizer = standardizing_dict[ncg]
        for i, snp in enumerate(snp_list):
            ncolors.append(COLORDIC[stdizer[i][snp]])
            nhatch.append(HATCHDIC[stdizer[i][snp]])
            nalpha.append(ALPHADIC[stdizer[i][snp]])
        #     nalleles.append(snp.upper())



    if not biggie:
        # # chloroplast genes
        ax, piechart = coffeetime(ax,
                                  fractions=cpfracs,
                                  size=1-SIZE,
                                  colors=cpolors,
                                  width=SIZE/2,
                                  lw=0,
                                  alpha=ALPH)
        piechart = hatchcake(piechart, calpha, chatch)
        ax, _ = coffeetime(ax,
                           fractions=cpfracs,
                           size=1-SIZE,
                           colors=['none']*len(cpfracs),
                           width=SIZE/2,
                           lw=1,
                           alpha=ALPH)

        # nucleus genes
        ax, piechart = coffeetime(ax,
                                  fractions=ncfracs,
                                  size=1,
                                  colors=ncolors,
                                  width=SIZE,
                                  lw=0,
                                  alpha=ALPH,
                                  startangle=90)
        piechart = hatchcake(piechart, nalpha, nhatch)
        ax, _ = coffeetime(ax,
                           fractions=ncfracs,
                           size=1,
                           colors=['none']*len(ncolors),
                           width=SIZE,
                           lw=1,
                           alpha=ALPH,
                           startangle=90)

    else:
        # chloroplast genes
        ax, piechart = coffeetime(ax,
                                  fractions=cpfracs,
                                  size=1-SIZE,
                                  colors=cpolors,
                                  width=SIZE,
                                  lw=0,
                                  alpha=ALPH/1.5)
        piechart = hatchcake(piechart, calpha, chatch)

        ax, _ = coffeetime(ax,
                           fractions=cpfracs,
                           size=1-SIZE,
                           colors=['none']*len(cpolors),
                           width=SIZE,
                           lw=1,
                           alpha=ALPH,
                           labels=calleles,
                           labeldistance=SIZE,
                           textprops=dict(size=6,
                                          color='k',
                                          horizontalalignment='center'))

        # nucleus genes
        ax, piechart = coffeetime(ax,
                                  fractions=ncfracs,
                                  size=1,
                                  colors=ncolors,
                                  width=SIZE,
                                  lw=0,
                                  alpha=ALPH/1.5,
                                  startangle=90)
        # hatch
        piechart = hatchcake(piechart, nalpha, nhatch)
        ax, _ = coffeetime(ax,
                           fractions=ncfracs,
                           size=1,
                           colors=['none']*len(ncolors),
                           width=SIZE,
                           lw=1,
                           alpha=ALPH,
                           labels=nalleles,
                           labeldistance=1-SIZE/2,
                           textprops=dict(size=6,
                                          color='k',
                                          horizontalalignment='center'),
                           startangle=90)

    return fig


# this function is core to use in different packages
def pie_the_indiv(meta_file, individual, figure, coords, biggie=False):
    '''
    Parameters
    ----------
    meta_file : summary file of the genetic files
    individual : which individual should be plotted? 1-8 possible

    Returns
    -------
    ax : figure axis

    '''
    meta_data = load_matrix(meta_file, infer_header=True, first_col=True)
    meta_data.index = [nm.replace('.fasta', '').replace('scent_primula_', '')
                       for nm in meta_data['file']]
    alignments = aligns(meta_file, meta_data)
    ax = pie_time(alignments,
                  meta_data,
                  individual,
                  figure,
                  coords,
                  biggie=biggie)

    return ax
