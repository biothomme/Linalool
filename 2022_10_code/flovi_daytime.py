#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 16:03:52 2020

@author: Thomsn
"""
# imports:
import numpy as np
import matplotlib as mpl


# functions:
def load_arguments():
    '''
    this is the argparse function

    it loads the inputfile (csv) of the flower visitors
    -------
    great movies!
    '''
    import argparse as ap

    parser = ap.ArgumentParser(description=__doc__,
                               formatter_class=ap.RawDescriptionHelpFormatter)
    parser.add_argument("-i",
                        action="store",
                        dest="csv_file",
                        required=True,
                        help="Input a csv_file with flower visitor data.")
    parser.add_argument("-m",
                        action="store",
                        dest="meta_file",
                        required=True,
                        help="Optional meta_data.")
    parser.add_argument("-g",
                        action="store",
                        dest="genus",
                        required=True,
                        help="offer a given genus")

    args = parser.parse_args()

    return args.csv_file, args.meta_file, args.genus


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

    # csv_data.fillna(value=None, inplace=True)

    return csv_data


# select only columns for the given genus:
def genuselect(flovi_data, genus):
    genus_key = genus[0].upper()
    sel_indx = [inx for inx in flovi_data.index if inx[1].upper() == genus_key]
    new_flovi_data = flovi_data.copy().loc[sel_indx, :]

    return new_flovi_data
    

def time_bins(start_time, end_time):
    import datetime

    MIDNIGHT = datetime.datetime.strptime('00:00', '%H:%M')
    BIN_MINUTES = 60
    MINS_A_DAY = 60*24

    all_bins = [MIDNIGHT + datetime.timedelta(minutes=BIN_MINUTES*i)
                for i in range(int(MINS_A_DAY/BIN_MINUTES))]
    time_dict = {tm: i for i, tm in enumerate(all_bins)}

    timerange = [time_dict[bi] for bi in all_bins
                 if (start_time - bi < datetime.timedelta(minutes=BIN_MINUTES) and
                     end_time - bi > datetime.timedelta(minutes=0))]

    return timerange, time_dict


# check if same species at same time and place already in frame:
def check_double(all_events, new_line):
    time = new_line[0]
    plotk = new_line[3]
    djumps = new_line[4]
    fintax = new_line[-1]
    sel_events = all_events[all_events['time']==time]
    if len(sel_events) > 0:
        sel_events = sel_events[sel_events['plot']==plotk]
        if len(sel_events) > 0:
            sel_events = sel_events[sel_events['djumps']==djumps]
            if len(sel_events) > 0:
                sel_events = sel_events[sel_events['fintax']==fintax]
                if len(sel_events) > 0:
                    return False

    return True
    

# get all timewindows a flo vi was present in
def time_windows(flower_visitor_data,
                 seperate_ants=True,
                 seperate_buflys=False):
    import pandas as pd
    import datetime

    ANTS = 'Formicidae'
    PAPILIONOIDEA = ['Nymphalidae',
                     'Pieridae',
                     'Papilionidae',
                     'Hesperiidae',
                     'Riodinidae',
                     'Lycaenidae',
                     'Hedylidae']

    flowvi_data = flower_visitor_data.copy()

    if seperate_ants:
        if ANTS in list(flowvi_data['Family'].values):
            ant_spots = flowvi_data.index[flowvi_data['Family'] == ANTS]
            for plott in ant_spots:
                flowvi_data.loc[plott, 'Order'] = ANTS
    if seperate_buflys:
        truth = flowvi_data['Family'].isin(PAPILIONOIDEA)
        if truth.shape[0] > 0:
            bufl_spots = flowvi_data.index[truth]
            for plott in bufl_spots:
                flowvi_data.loc[plott, 'Order'] = 'Papilionoidea'

    all_taxa = [tx for tx in set(list(flowvi_data['Order'].values))
                if isinstance(tx, str)]

    all_events = pd.DataFrame(columns=['time',
                                       'taxon',
                                       'abundance',
                                       'plot',
                                       'djumps',
                                       'fintax'])
    for i, row in flowvi_data.iterrows():
        start_time = datetime.datetime.strptime(row['sTime'], '%H:%M')
        end_time = datetime.datetime.strptime(row['eTime'], '%H:%M')
        timerange, time_dict = time_bins(start_time, end_time)
        for timepoint in timerange:
            fintax = f'{row["Family"]}-{row["Genus"]}-{row["Species"]}'
            
            if not ((pd.isna(row['Order'])) or row['Confidence'] == 'low'):
                new_line = [timepoint,
                            row['Order'],
                            row['maxAbund'],
                            row['Plot_key'],
                            row['PolDjumps'],
                            fintax]
                if check_double(all_events, new_line):
                    all_events.loc[len(list(all_events.index)),
                                   :] = new_line

    return all_events, time_dict


# obtain a overall color_dictionary forall taxa
def col_dic_all_taxa(full_event_frame):
    from matplotlib import cm

    COLOR_MAP = 'tab20'

    cmap = cm.get_cmap(COLOR_MAP, 256)
    colo = full_event_frame['taxon'].unique()

    col_dic = dict()
    for i, tax in enumerate(colo):
        col = cmap(i/len(colo))
        col_dic[tax] = col

    return col_dic


# obtain the camera time lines
def so_meta_oida(meta_data, event_frame):
    import datetime
    import pandas as pd

    camera_plots = meta_data['camera'].dropna().index
    cameta_data = meta_data.loc[camera_plots, :].copy()

    cameta_data['species'] = [ev[1:3] for ev in cameta_data.index]
    event_frame['species'] = [ev[1:3] for ev in event_frame['plot']]

    species = [sp for sp in event_frame['species'].unique() if 'x' not in sp]
    sp_bool = [row.name for i, row in cameta_data.iterrows()
               if row['species'] in species]
    cameta_data = cameta_data.loc[sp_bool, :].copy()

    camtime_frame = pd.DataFrame(columns=['time',
                                          'taxon',
                                          'abundance',
                                          'plot',
                                          'djumps',
                                          'fintax'])
    for i, row in cameta_data.iterrows():
        for j in range(int(row['djumps'])+1):
            st_clm = str(row[f'on_eff_d{1+j}'])
            en_clm = str(row[f'off_eff_d{1+j}'])
            if (st_clm != 'nan' and en_clm != 'nan'):
                start_time = datetime.datetime.strptime(st_clm, '%H:%M')
                end_time = datetime.datetime.strptime(en_clm, '%H:%M')
                timerange, _ = time_bins(start_time, end_time)
                for timepoint in timerange:
                    camtime_frame.loc[len(list(camtime_frame.index)),
                                      :] = [timepoint,
                                            row['species'],
                                            row['pspecimen_nr'],
                                            row.name,
                                            row['djumps'],
                                            row['species']]
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', -1)

    return camtime_frame


# make plotting_frame for histogram
def ploffo(event_frame, time_dict):
    import pandas as pd

    colo = np.sort(event_frame['taxon'].unique())
    timo = time_dict.values()
    plotting_frame = pd.DataFrame(np.zeros([len(timo),
                                           len(colo)]),
                                  columns=colo,
                                  index=timo)
    for i, row in event_frame.iterrows():
        tax = row['taxon']
        tim = row['time']
        plotting_frame.loc[tim, tax] += int(row['abundance']) / int(row['abundance'])

    return plotting_frame, colo, timo


# standardize visits by cameraplots
def plot_std_by_cams(cam_setup, plotting_frame):
    cam_setup = cam_setup.astype(int)
    plotting_frame = plotting_frame.div(cam_setup.values, fill_value=0).copy()
    plotting_frame.replace(np.nan, 0, inplace=True)
    plotting_frame.replace(np.inf, 0, inplace=True)
    
    return plotting_frame.copy()


# plot a histogram of the flower_visit timeline
def historical(event_frame,
               time_dict,
               ax,
               col_dic,
               species,
               cam_setup,
               meta_data=None,
               min_frame=None):
    import datetime
    import pandas as pd

    # assumption: sunrise time: 4:00 - 6:00 am / sunset time: 8:00 - 10:00 pm
    SUNRISE = ['03:00', '06:00']
    SUNSET = ['20:00', '23:00']
    SPLITA = 50
    
    sunrise = [time_dict[datetime.datetime.strptime(sr, '%H:%M')] for
               sr in SUNRISE]
    sunset = [time_dict[datetime.datetime.strptime(sr, '%H:%M')] for
              sr in SUNSET]

    plotting_frame, colo, timo = ploffo(event_frame, time_dict)
    plotting_frame = plotting_frame.astype(int)

    cam_setup = cam_setup.astype(int)
    plotting_frame = plot_std_by_cams(cam_setup, plotting_frame)

    if not isinstance(meta_data, type(None)):
        abundance_frame, geo_frame, spec_frame = abundanza(event_frame,
                                                           time_dict,
                                                           meta_data)

    # background
    ax.axvspan(min(time_dict.values()),
               sunrise[0],
               facecolor='grey',
               alpha=1)
    ax.axvspan(sunset[1],
               max(time_dict.values())+1,
               facecolor='grey',
               alpha=1)
    particle = (sunrise[1] - sunrise[0]) / SPLITA
    for i in range(SPLITA):
        ax.axvspan(sunrise[0] + particle*(i-0.001),
                   sunrise[0] + particle*(i+1.001),
                   facecolor='grey',
                   alpha=1 - i/SPLITA)
        ax.axvspan(sunset[0] + particle*(i-0.001),
                   sunset[0] + particle*(i+1.001),
                   facecolor='grey',
                   alpha=i/SPLITA)

    bars = np.zeros(len(timo))
    for tax in colo:
        col = col_dic[tax]
        ax.bar(np.array(list(timo))+.5,
               plotting_frame[tax],
               bottom=bars,
               edgecolor='white',
               width=1,
               label=tax,
               color=col,
               lw=.5)
        if not isinstance(min_frame, type(None)):
            ax.bar(np.array(list(timo))+.5,
                   min_frame[tax],
                   bottom=bars,
                   edgecolor=col,
                   width=1,
                   color='white',
                   hatch='\\\\\\\\\\\\\\',
                   alpha=.5,
                   lw=0)
        if not isinstance(meta_data, type(None)):
            ab = abundance_frame[tax]
            ge = geo_frame[tax]
            sp = spec_frame[tax]
            for i, brr in enumerate(bars):
                if plotting_frame[tax].iloc[i] > 2/cam_setup.values[i]:
                    txt = f'{ab.iloc[i]:.1f}\n{sp.iloc[i]}|{ge.iloc[i]}'
                    ax.text(s=txt,
                            x=i+.5,
                            y=brr+(.5*plotting_frame[tax].iloc[i]),
                            size=4,
                            ha='center',
                            va='center')
        bars = np.add(bars, plotting_frame[tax])

    # for i, cam_number in enumerate(cam_setup.values):
    #     ax.text(s=f'{int(cam_number)}',
    #             x=i+.5,
    #             y=bars[i]+(.05*max(bars)),
    #             size=6,
    #             ha='center')
    ax.legend(loc='right')

    return ax

# seperate the total event frame into two for the two species
def sepevents(event_frame):
    event_frame['species'] = [ev[1:3] for ev in event_frame['plot']]
    species = [sp for sp in event_frame['species'].unique() if 'x' not in sp]
    event_splits = dict()

    event_split1 = event_frame.loc[event_frame['species'] == species[0], :]
    event_split2 = event_frame.loc[event_frame['species'] == species[1], :]
    
    event_splits[species[0]] = event_split1.copy()
    event_splits[species[1]] = event_split2.copy()

    return event_splits


# make small heatbar for camera setup
def heatbar(camera_setup, axis, xlimits):
    limits = [0, len(camera_setup), 0, 2]
    axis.imshow(camera_setup.transpose(),
                cmap='binary',
                aspect='auto',
                extent=limits)

    camera_setup = camera_setup.transpose()
    camera_setup = camera_setup.reindex(index=camera_setup.index[::-1]).transpose()

    max_cams = np.max(camera_setup.values)

    for i, row in camera_setup.iterrows():
        for j, cams in enumerate(row):
            if int(cams) > max_cams/2:
                col = 'lightgrey'
            else:
                col = 'dimgrey'

            axis.text(i+.5,
                      j+.5,
                      str(int(cams)),
                      ha='center',
                      va='center',
                      c=col,
                      size=8)

    for edge, spine in axis.spines.items():
        spine.set_visible(False)

    axis.set_xticks(np.arange(camera_setup.shape[0]+1)-.001, minor=True)
    axis.set_yticks(np.arange(camera_setup.shape[1]+1)-.001, minor=True)
    axis.grid(which='minor', color='w', linestyle='-', linewidth=1)
    axis.tick_params(which='minor', bottom=False, left=False)

    axis.set_xlim(xlimits)

    axis.set_xticklabels([])
    axis.set_yticklabels([])

    return axis


# obtain the minimum occurences of flovi per timepoint between both taxa
def minimal(event_frames, camtime_frames):
    import pandas as pd

    tot_frame = pd.concat(list(event_frames.values()))
    sps = list(event_frames.keys())
    plotting_frame1, _, _ = ploffo(event_frames[sps[0]], time_dict)
    plotting_frame2, _, _ = ploffo(event_frames[sps[1]], time_dict)

    cam_setup1, _, _ = ploffo(camtime_frames[sps[0]], time_dict)
    cam_setup2, _, _ = ploffo(camtime_frames[sps[1]], time_dict)

    plotting_frame1 = plot_std_by_cams(cam_setup1, plotting_frame1)
    plotting_frame2 = plot_std_by_cams(cam_setup2, plotting_frame2)

    min_frame = pd.DataFrame(index=plotting_frame1.index,
                             columns=tot_frame['taxon'].unique())
    for (i, row), (j, qow) in zip(plotting_frame1.iterrows(),
                                  plotting_frame2.iterrows()):

        for tax in min_frame.columns:
            min_value = 0
            if (tax in row.index and tax in qow.index):
                min_value = min([row.loc[tax], qow.loc[tax]])
            min_frame.loc[i, tax] = min_value

    return min_frame


# make a frame summarizing the mean abundance of a taxon per plot
def abundanza(event_frame, time_dict, meta_data):
    import pandas as pd

    plotting_frame, _, _ = ploffo(event_frame, time_dict)

    abundance_frame = pd.DataFrame(columns=plotting_frame.columns,
                                   index=plotting_frame.index)
    geo_frame = abundance_frame.copy()
    spec_frame = abundance_frame.copy()

    for time in abundance_frame.index:
        abtime_events = event_frame.loc[event_frame['time'] == time]
        for tax in abundance_frame.columns:
            # abundance_frame
            abtaxtime_events = abtime_events.loc[abtime_events['taxon'] == tax]
            occurences = abtaxtime_events.shape[0]

            total_abundance = np.sum(abtaxtime_events['abundance'].values)
            if total_abundance > 0:
                abundance_frame.loc[time, tax] = total_abundance / occurences
            else:
                abundance_frame.loc[time, tax] = 0

            # geo_frame
            all_plots = list(abtaxtime_events['plot'].values)
            all_places = [row['place_key'] for inx, row in meta_data.iterrows()
                          if inx in all_plots]
            all_places = list(set([pl[:3] for pl in all_places]))
            geo_frame.loc[time, tax] = len(all_places)

            # species_frame
            all_specs = list(abtaxtime_events['fintax'].values)

            spec_frame.loc[time, tax] = len(list(set(all_specs)))

    return abundance_frame, geo_frame, spec_frame


# helper function to calculate cross pollination probabilities and stor them.
def cross_pollination_probablity(event_frame, time_dict, sp1, sp2, file_name):
    import pandas as pd
    plotting_frame, _, _ = ploffo(event_frame[sp1], time_dict)
    diurnal_visits = (plotting_frame.sum(axis=0))
    diurnal_frequencies = diurnal_visits / diurnal_visits.sum()
    plotting_frame, _, _ = ploffo(event_frame[sp2], time_dict)
    diurnal_visits = (plotting_frame.sum(axis=0))
    diurnal_frequencies2 = diurnal_visits / diurnal_visits.sum()
    di_freq = pd.DataFrame([diurnal_frequencies, diurnal_frequencies2], index=[sp1, sp2]).fillna(0).transpose()
    overall_frequencies = di_freq.sum(axis=1)
    overall_visit_probabilities = overall_frequencies/overall_frequencies.sum()
    total_pollinations = di_freq.apply(lambda x: x[0]**2 + 2*x[0]*x[1] + x[1]**2, axis=1)
    cross_pollinations = di_freq.apply(lambda x: 2*x[0]*x[1], axis=1)
    cross_pollination_probability = cross_pollinations / total_pollinations * overall_visit_probabilities
    cpp = cross_pollination_probability.sum()
    with open(file_name, 'w') as of:
        of.write(f"cross pollination probability\n{'/'.join([sp1, sp2])}\t{abs(cpp):.3f}\n\n")
        of.write(f"cross pollination disproportion\n{'/'.join([sp1, sp2])}\t{abs(cpp/(1-cpp)):.3f}\n\n")
    return


# plot the whole timeline
def plot_two_axes_symme(csv_file,
                        genus,
                        event_frames,
                        time_dict,
                        col_dict,
                        camtime_frames,
                        meta_data=None,
                        min_frame=None):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import pandas as pd

    oldout_file = csv_file.split('/')
    out_file = '/'.join(oldout_file[:-1] + [f'flovi_timeline_{genus}.pdf'])
    out_cpp_file = '/'.join(oldout_file[:-1] + [f'flovi_timeline_{genus}_crosspollination_probs.txt'])


    reverse_time = {point: time.strftime('%H:%M') for time, point
                    in time_dict.items()}
    sp1 = list(event_frames.keys())[0]
    sp2 = list(event_frames.keys())[1]

    cam_setup1, _, _ = ploffo(camtime_frames[sp1], time_dict)
    cam_setup2, _, _ = ploffo(camtime_frames[sp2], time_dict)

    fig = plt.figure(figsize=(8, 6))
    gridspec.GridSpec(16, 1, hspace=0)

    ax1 = plt.subplot2grid((16, 1), (0, 0), rowspan=7)
    hm1 = plt.subplot2grid((16, 1), (7, 0), rowspan=2)
    ax2 = plt.subplot2grid((16, 1), (9, 0), rowspan=7)

    cross_pollination_probablity(event_frames, time_dict, sp1, sp2, out_cpp_file)

    ax1 = historical(event_frames[sp1],
                     time_dict,
                     ax1,
                     col_dict,
                     sp1,
                     cam_setup1,
                     meta_data=meta_data,
                     min_frame=min_frame)
    xlims = ax1.get_xlim()

    cam_setotal = pd.concat([cam_setup1, cam_setup2], axis=1)
    hm1 = heatbar(cam_setotal, hm1, xlims)

    ax2 = historical(event_frames[sp2],
                     time_dict,
                     ax2,
                     col_dict,
                     sp2,
                     cam_setup2,
                     meta_data=meta_data,
                     min_frame=min_frame)

    fig.subplots_adjust(hspace=0)

    xlocs = list(range(0, len(time_dict)+1, 4))
    xlab = [reverse_time[tick] if tick in reverse_time.keys()
            else '' for tick in xlocs]
    xlab = ['24:00' if (lab == '' and xlab[i-1] != '')else lab for i, lab
            in enumerate(xlab)]

    ax1.set_xticks([])
    ax1.set_xticklabels([])
    ax2.set_xticks(xlocs)
    ax2.set_xticklabels(xlab)

    ax1.set_xlim([min(time_dict.values()), max(time_dict.values())+1])
    hm1.set_xlim([min(time_dict.values()), max(time_dict.values())+1])
    ax2.set_xlim([min(time_dict.values()), max(time_dict.values())+1])

    ylims = max([ax1.get_ylim(), ax2.get_ylim()])
    ax1.set_ylim(ylims)

    ax2.set_ylim(ylims)

    ax2.invert_yaxis()

    ax1.set_ylabel(f'average flower visits per seperate taxonomic unit / {sp1}')
    ax2.set_ylabel(f'{sp2}')
    ax2.set_xlabel('daytime')

    fig.savefig(out_file)

    return


# constants:


# main:
mpl.rcParams['pdf.fonttype'] = 42

csv_file, meta_file, genus = load_arguments()
flovi_data = load_matrix(csv_file, infer_header=True, first_col=False)
flovi_data = genuselect(flovi_data, genus)
meta_data = load_matrix(meta_file, infer_header=True)

if genus[0].lower() == 'p':
    ev_frame, time_dict = time_windows(flovi_data,
                                       seperate_ants=True,
                                       seperate_buflys=True)
else:
    ev_frame, time_dict = time_windows(flovi_data,
                                       seperate_ants=True)

event_splits = sepevents(ev_frame)
camera_time_frame = so_meta_oida(meta_data, ev_frame)
camera_time_splits = sepevents(camera_time_frame)

col_dic = col_dic_all_taxa(ev_frame)

min_frame = minimal(event_splits, camera_time_splits)

plot_two_axes_symme(csv_file,
                    genus,
                    event_splits,
                    time_dict,
                    col_dic,
                    camera_time_splits,
                    meta_data=meta_data,
                    min_frame=min_frame)
