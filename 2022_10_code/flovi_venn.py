#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 08:54:39 2020

@author: Thomsn
"""
# pip install matplotlib-venn-wordcloud
# pip install matplotlib-venn

# imports:
import numpy as np
import matplotlib as mpl


# functions:
def load_arguments():
    '''
    this is the argparse function

    it loads the inputfile (csv) of the flower visitors
    -------
    great animals!
    '''
    import argparse as ap

    parser = ap.ArgumentParser(description=__doc__,
                               formatter_class=ap.RawDescriptionHelpFormatter)
    parser.add_argument("-i",
                        action="store",
                        dest="csv_file",
                        required=True,
                        help="Input a csv_file with flower vis data (species).")
    parser.add_argument("-m",
                        action="store",
                        dest="meta_file",
                        required=False,
                        help="Optional meta_data.")
    parser.add_argument("-g",
                        action="store",
                        dest="genus",
                        required=False,
                        help="(p)rimula,(g)entiana,(r)hododendron or (s)ilene.")
    parser.add_argument("-l",
                        action="store",
                        dest="loca",
                        required=False,
                        help="file containing meta locations")

    args = parser.parse_args()

    return args.csv_file, args.meta_file, args.genus, args.loca


def load_matrix(csv_file, infer_header=False, sep=';', first_col=False):
    '''
    Parameters
    ----------
    csv_file : containing the minimal data file/data

    Returns a pd.DataFrame of it
    -------
    '''
    import pandas as pd

    if infer_header:
        csv_data = pd.read_csv(csv_file, sep=sep)
    else:
        csv_data = pd.read_csv(csv_file, sep=sep, header=None)

    if not first_col:
        csv_data.index = csv_data.iloc[:, 0].values
        csv_data = csv_data.iloc[:, 1:].copy()

    return csv_data


def genimize(flovi_data, genus):
    PRIMULAE = {'Pl': 'Primula lutea',
                'Pa': 'Primula lutea',
                'Ph': 'Primula hirsuta'} # ,
                # 'Px': 'Primula lutea x hirsuta'}
    RHODIS = {'Rf': 'Rhododendron ferrugineum',
              'Rh': 'Rhododendron hirsutum'} # ,
            #   'Rx': 'Rhododendron ferrugineum x hirsutum'}
    GENTIANAE = {'Ga': 'Gentiana acaulis',
                 'Gc': 'Gentiana clusii'} #,
                 #'Gx': 'Gentiana acaulis x clusii'}
    SILENE = {'Sa': 'Silene acaulis subsp. acaulis',
              'Sa': 'Silene acaulis subsp. bryoides'}

    genus = genus.lower()[0]
    if genus == 'g':
        plant = GENTIANAE
    elif genus == 'r':
        plant = RHODIS
    elif genus == 's':
        plant = SILENE
    else:
        plant = PRIMULAE
    genus_list = map(lambda x: (x[1].lower()==genus and 'x' not in x), flovi_data['plot_key'])

    genus_data = flovi_data.loc[list(genus_list), :].copy()
    genus_data['plant'] = list(map(lambda x: plant[x[1:3]],
                                   genus_data['plot_key']))
    return genus_data


def locate_taxa(flovi_data, meta_data, loc_data):
    import pandas as pd
    
    # lambdas
    isnan = lambda x: not (x != x)
    sticktogether = lambda x_tuple: ' '.join(x_tuple) if isnan(x_tuple[1]) else\
        (f'{x_tuple[0]} sp.'if isnan(x_tuple[0]) else 'indet')

    # main
    species = [pt for pt in flovi_data['plant'].unique() if not ' x ' in pt]
    sp = [pt.replace(' ', '_').lower() for pt in species]
    flovi_data['taxon'] = list(map(sticktogether, zip(flovi_data['genus'],
                                                      flovi_data['species'])))
    tax_frame = pd.DataFrame(index=flovi_data['taxon'].unique(),
                             columns=[f'{sp[0]}_list',
                                      f'{sp[0]}_num',
                                      f'{sp[1]}_list',
                                      f'{sp[1]}_num'])

    totax_tuples = []
    for spc in sp:
        for place in set([m[:3] for m in meta_data['place_key'].unique()]):
            totax_tuples.append((spc, place))

    zers = np.zeros((len(flovi_data['taxon'].unique()), len(totax_tuples)))
    totax_frame = pd.DataFrame(zers,
                               index=flovi_data['taxon'].unique(),
                               columns=pd.MultiIndex.from_tuples(totax_tuples))

    sp_dict = {spc: [] for spc in sp}
    
    for spec in species:
        spc = spec.replace(' ', '_').lower()
        spe_data = flovi_data.loc[flovi_data['plant'] == spec, :]

        for tax in tax_frame.index:
            tax_data = spe_data.loc[spe_data['taxon'] == tax, :]
            plots = tax_data['plot_key']

            loc_list = []
            for plot in plots:
                plt = meta_data.loc[plot, 'place_key'][:3]
                totax_frame.loc[tax, (spc, plt)] += 1

                if plt not in loc_list:
                    loc_list.append(plt)
                    sp_dict[spc].append(tax)
            tax_frame.loc[tax, f'{spc}_list'] = ', '.join(sorted(loc_list))
            tax_frame.loc[tax, f'{spc}_num'] = len(loc_list)

    # remove empty columns
    totax_sum = totax_frame.sum(axis=0)
    totax_frame = totax_frame.loc[:, totax_sum != 0]

    return tax_frame, totax_frame, sp_dict


# make multiindex for higher taxonomic units of flovis and sort by
def increase_taxframe(tax_frame, flovi_data, remove_indet=True):
    import pandas as pd

    if remove_indet:
        tax_frame = tax_frame.loc[tax_frame.index != 'indet', :].copy()

    new_inds = []
    for tax in tax_frame.index:
        sel_frame = flovi_data.loc[flovi_data['taxon'] == tax, :]
        new_ind = (sel_frame['order'].values[0], tax)
        new_inds.append(new_ind)

    tax_frame.index = pd.MultiIndex.from_tuples(new_inds)
    tax_frame.sort_index(inplace=True)
    return tax_frame


# retrieve points to set a line (taxon seperator)
def ablines(totaltax_frame):
    x_y_axes = {}
    for ax, axlist in zip(['x', 'y'],
                          [totaltax_frame.columns, totaltax_frame.index]):
        axaxis = [tt[0] for tt in axlist]
        axlines = [i+1 for i in range(len(axaxis)-1)
                   if axaxis[i] != axaxis[i+1]]
        # axlines = [li/len(axaxis) for li in axlines]
        axlines = [li for li in axlines]
        x_y_axes[ax] = axlines

    return x_y_axes


def totalix(totaltax_frame):
    import pandas as pd

    specs = totaltax_frame.columns.get_level_values(0).unique()
    sum_frame = totaltax_frame.copy()
    sum_frame[sum_frame > 0] = 1
    taxsum = pd.DataFrame(index=totaltax_frame.index,
                          columns=[(sp, 'sum') for sp in specs])
    for sp in specs:
        taxsum[(sp, 'sum')] = sum_frame[sp].sum(axis=1)

    factor = totaltax_frame.max().max() / taxsum.max().max()
    taxsum = taxsum.multiply(factor)

    newtax_frame = totaltax_frame[specs[0]].sort_index(axis=1).copy()
    new_tax_col = [(specs[0], tt) for tt in totaltax_frame[specs[0]].columns]

    newtax_frame = pd.concat([newtax_frame, taxsum], axis=1)
    new_tax_col += list(taxsum.columns)
    newtax_frame = pd.concat([newtax_frame,
                              totaltax_frame[specs[1]].sort_index(axis=1).copy()],
                             axis=1)
    new_tax_col += [(specs[1], tt) for tt in totaltax_frame[specs[1]].columns]

    newtax_frame.columns = pd.MultiIndex.from_tuples(new_tax_col)

    return newtax_frame.copy()


def plotoplot(csv_file, genus, totaltax_frame, tax_frame, sp_dict):
    import matplotlib.pyplot as plt
    from matplotlib_venn import venn2
    # from matplotlib_venn_wordcloud import venn2_wordcloud

    out_file = csv_file.replace('.csv', f'_{genus}_hmp.pdf')
    out_div_file = csv_file.replace('.csv', f'_{genus}_diversity_measures.txt')

    fig = plt.figure(figsize=(8, 6))

    # store biodiv measures
    compute_diversity_indices(totaltax_frame, out_div_file)

    # heatmap
    hm_frame = totalix(totaltax_frame)

    hm_frame.to_csv(f"{genus}_flovi_region_binned.csv")
    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(hm_frame,
               cmap='binary',
               aspect='auto')

    tax_lines = ablines(hm_frame)

    for axa, lines in tax_lines.items():
        for line in lines:
            if axa == 'x':
                ax1.axvline(line-.5, lw=1, color='dimgrey')
                ax1.axvline(line+.5, lw=1, color='black')
                ax1.axvline(line-1.5, lw=1, color='black')
            else:
                ax1.axhline(line-.5, lw=1, color='dimgrey')

    ax1.set_xticks(np.arange(hm_frame.shape[1]))
    ax1.set_yticks(np.arange(hm_frame.shape[0]))
    ax1.set_xticklabels([tt[1] for tt in hm_frame.columns])
    if isinstance(hm_frame.index[0], tuple):
        ax1.set_yticklabels([tt[-1] for tt in hm_frame.index])
    else:
        ax1.set_yticklabels(hm_frame.index)

    ax11 = ax1.twiny()
    ax11.patch.set_visible(False)
    ax11.yaxis.set_visible(False)
    for spinename, spine in ax11.spines.items():
        if spinename != 'top':
            spine.set_visible(False)
    ax11.spines['top'].set_position(('outward', 30))

    speci_ticks = [i for i, cl in enumerate(hm_frame.columns) if
                   (i > 0 and hm_frame.columns[i-1][0] != cl[0])]
    spec_ticks = [0] + [spt / len(hm_frame.columns) for spt in speci_ticks] + [1]
    ax11.set_xticks(spec_ticks, minor=False)
    ax11.set_xticklabels([], minor=False)
    speci_ticks.append(len(hm_frame.columns))
    spec_names = {hm_frame.columns[speci_ticks[i]-1][0].split('_')[1]:
                  np.mean(spec_ticks[i:i+2])
                  for i in np.arange(len(spec_ticks)-1)}

    ax11.set_xticks(list(spec_names.values()), minor=True)
    ax11.set_xticklabels(list(spec_names.keys()), minor=True)

    for tick in ax11.xaxis.get_minor_ticks():
        tick.tick1line.set_markersize(0)
        tick.tick2line.set_markersize(0)
        tick.label1.set_horizontalalignment('center')

    ax11.tick_params(axis='x', direction='in')
    ax1.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    ax1.set_xticklabels(labels=ax1.get_xticklabels(), rotation=-30)# , rotation=-30, ha="right",
              #rotation_mode="anchor")

    # venn
    ax2 = plt.subplot(1, 2, 2)
    sets = [set(val) for val in sp_dict.values()]
    set_labels = [val for val in sp_dict.keys()]

    venn2(subsets=sets,
          set_labels=set_labels,
          ax=ax2)
    # fosize = [se for sl in sets for se in sl]
    # fosize = {se: sum([1 for val in sp_dict.values() for sp in val
    #                    if sp == se])+1 for se in fosize}
    # venn2_wordcloud(sets=sets,
    #                 ax=ax,
    #                 word_to_frequency=fosize,
    #                 wordcloud_kwargs={'min_font_size': 5,
    #                                   'font_step':3})

    fig.savefig(out_file, bbox_inches='tight')

    return


def compute_diversity_indices(totaltax_frame, file_name):
    """Obtain alpha diversity and bray curtis of flower visitor communites between both plant species

    Args:
        totaltax_frame (pandas.DataFrame): Dataframe summarizing abundances of flower visitors for a given species at a specific plot.
    """
    tax_spec_ab_frame = totalix(totaltax_frame)
    with open(file_name, 'w') as of:
        of.write("-- overall --\n")
    alpha_beta_div(tax_spec_ab_frame, file_name)
    for indx, taxon_frame in tax_spec_ab_frame.groupby(level=0, axis=0):
        with open(file_name, 'a') as of:
            of.write(f"{taxon_frame.index[0][0].lower()}\n")
        alpha_beta_div(taxon_frame, file_name)
    cross_pollination_frequency(tax_spec_ab_frame, file_name)
    return

# help function to compute alpha and beta diversity measures for data(sub)sets
def alpha_beta_div(tax_spec_ab_frame, file_name):
    import pandas as pd
    import skbio.diversity as sbd
    ab_frame = pd.DataFrame(index=tax_spec_ab_frame.index)
    with open(file_name, 'a') as of:
        of.write(f"alpha diversity (shannon)\n")
        for indx, species in tax_spec_ab_frame.groupby(level=0, axis=1):
            sp = species.columns[0][0]
            abundances = species.loc[:, (sp, "sum")]
            ab_sum = sum(abundances)
            if ab_sum != 0:
                ab_frame[sp] = abundances / sum(abundances)
                alpha_div = sbd.alpha_diversity("shannon", abundances)
            else:
                ab_frame[sp] = abundances
                alpha_div = [0]
            of.write(f"{sp}\t{abs(alpha_div[0]):.3f}\n")
        beta_div = sbd.beta_diversity("braycurtis", ab_frame.transpose(), ids=ab_frame.columns)
        of.write(f"beta diversity (bray curtis)\n{'/'.join(ab_frame.columns)}\t{abs(beta_div[1, 0]):.3f}\n\n")
    return

# helper function to compute probabilities of cross pollination
def cross_pollination_frequency(tax_spec_ab_frame, file_name):
    with open(file_name, 'a') as of:
        of.write(f"\n-- cross-pollination probability --\n")
        for i, level in zip([0, 1], ["order", "species"]):
            abundances = tax_spec_ab_frame.loc[:, (slice(None), "sum")].groupby(level=i).sum()
            species = [col[0] for col in abundances.columns]
            visit_frequencies = abundances / abundances.sum()
            overall_abundances = abundances.sum(axis=1)
            overall_visit_probabilities = overall_abundances/overall_abundances.sum()
            total_pollinations = visit_frequencies.apply(lambda x: x[0]**2 + 2*x[0]*x[1] + x[1]**2, axis=1)
            cross_pollinations = visit_frequencies.apply(lambda x: 2*x[0]*x[1], axis=1)
            cross_pollination_probability = cross_pollinations / total_pollinations * overall_visit_probabilities
            cpp = abs(cross_pollination_probability.sum())
            of.write(f"resolution level {level}\n{'/'.join(species)}\t{cpp:.3f}\n")
            of.write(f"cross pollination disproportion {'/'.join(species)}\t{abs(cpp/(1-cpp)):.3f}\n\n")
    return

# constants:


# main:
mpl.rcParams['pdf.fonttype'] = 42

csv_file, meta_file, genus, location_file = load_arguments()
flovi_data = load_matrix(csv_file, infer_header=True, first_col=False)
meta_data = load_matrix(meta_file, infer_header=True)
loc_data = load_matrix(location_file, infer_header=True)

flovi_data = genimize(flovi_data, genus)
print(flovi_data)
tax_frame, totaltax_frame, sp_dict = locate_taxa(flovi_data,
                                                 meta_data,
                                                 loc_data)

tax_frame = increase_taxframe(tax_frame, flovi_data)
totaltax_frame = increase_taxframe(totaltax_frame, flovi_data)

plotoplot(csv_file, genus, totaltax_frame, tax_frame, sp_dict)
