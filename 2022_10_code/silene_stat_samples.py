#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 13:11:59 2020

@author: Thomsn
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plant_genus = 'silene'
num = 11

os.chdir('/Users/Thomsn/Desktop/posterscent')

if plant_genus == 'silene':
    with open(f'sil_sam_dat.csv') as infofile:
        info_df = pd.read_csv(infofile, sep = ';')
        
    with open(f'sil_pop_dat.csv') as popfile:
        pop_df = pd.read_csv(popfile, sep = ';')


info_df.loc[:,'events'] = [f'{i[1]}_{str(i[2])[0]}' for i in info_df.loc[:,('KS', 'Scent')].itertuples()]
pop_names = np.sort(info_df.loc[:,'population'].unique())
event_names = np.sort(info_df.loc[:,'events'].unique())
sisu_df = pd.DataFrame(index = event_names, columns = pop_names)
sin_ser = pd.Series(index = pop_names)
width = 0.35       # the width of the bars: can also be len(x) sequence

N = len(pop_names) # number of populations
#menMeans = (20, 35, 30, 35, 27) # later
#womenMeans = (25, 32, 34, 20, 25) # later
ind = [i for i in np.arange(N)]    # the x locations for the groups

for population in pop_names:
    popat = info_df[(info_df.loc[:,'population'] == population)]
    sin_ser[population] = len(popat.index)
    for event in event_names:
        popev = popat[(popat.loc[:,'events'] == event)]
        sisu_df.at[event, population] = len(popev.index) / sin_ser[population]

xtick = [f'{pop_names[i]} \n({int(sin_ser.iat[i])})' for i, _ in enumerate(pop_names)]

fig = plt.figure()
ax = fig.add_subplot()

bottom = [0] * len(pop_names)
ratios = [.33, .54, .07, .06]
colors = ['dodgerblue', 'darkblue', 'salmon', 'red']
bars = pd.Series(index = event_names)

for j, event in enumerate(event_names):
    ratios = [i for i in sisu_df.loc[event]]
    height = ratios
    bars[event] = ax.bar(ind, height, width, bottom=bottom, color=colors[j])
    ypos = bottom + ax.patches[j].get_height() / 2
    bottom = [q+w for q,w in zip(height, bottom)]
   # ax.text(ind, ypos, "%d%%" % (ax.patches[j].get_height() * 100),
    #         ha='center')
    
plt.xlabel('Population')
plt.ylabel('Ratio')
plt.title('How successful was the analysis?')
plt.xticks(ind, xtick)
plt.legend(bars, 
           ('Calc. Soil - no scent', 'Calc. Soil - scent', 'Sil. Soil - no scent', 'Sil. Soil - no scent'),
           bbox_to_anchor=(1.05, .5), loc='center left', borderaxespad=0.)
plt.yticks(np.arange(0, 1, .1))
plt.savefig(f'figs/{plant_genus}_sampling_pop.pdf', format='pdf', bbox_inches='tight')
plt.show()


#### ----- ####
group_names = np.sort(pop_df.group.unique())
event_names = np.sort(info_df.loc[:,'events'].unique())
sigu_df = pd.DataFrame(index = event_names, columns = group_names)
sign_ser = pd.Series(index = group_names)
for group in group_names:
    pop_set = pop_df[pop_df.group == group].population
    sign_ser[group] = sum([int(sin_ser[x]) for x in sin_ser.keys() if any(pop_set == x)])
    for event in event_names:
        popev = sum([int(np.round(sisu_df.at[event, x]*sin_ser[x])) for x in pop_set])
        sigu_df.at[event, group] = popev / sign_ser[group]

pop_names = group_names
sisu_df = sigu_df
sin_ser = sign_ser
width = 0.35       # the width of the bars: can also be len(x) sequence

xtick = [f'{pop_names[i]} \n({int(sin_ser.iat[i])})' for i, _ in enumerate(pop_names)]

N = len(pop_names) # number of populations

ind = [i for i in np.arange(N)]    # the x locations for the groups

fig = plt.figure()
ax = fig.add_subplot()

bottom = [0] * len(pop_names)
ratios = [.33, .54, .07, .06]
colors = ['dodgerblue', 'darkblue', 'salmon', 'red']
bars = pd.Series(index = event_names)

for j, event in enumerate(event_names):
    ratios = [i for i in sisu_df.loc[event]]
    height = ratios
    bars[event] = ax.bar(ind, height, width, bottom=bottom, color=colors[j])
    ypos = bottom + ax.patches[j].get_height() / 2
    bottom = [q+w for q,w in zip(height, bottom)]
   # ax.text(ind, ypos, "%d%%" % (ax.patches[j].get_height() * 100),
    #         ha='center')
    
plt.xlabel('Population')
plt.ylabel('Ratio')
plt.title('Sampling success per population groups')
plt.xticks(ind, xtick)
plt.legend(bars, 
           ('Calc. Soil - no scent', 'Calc. Soil - scent', 'Sil. Soil - no scent', 'Sil. Soil - no scent'),
           bbox_to_anchor=(1.05, .5), loc='center left', borderaxespad=0.)
plt.yticks(np.arange(0, 1, .1))
plt.savefig(f'figs/{plant_genus}_sampling_group.pdf', format='pdf', bbox_inches='tight')
plt.show()