#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 18:05:48 2020

that great bunch of code is thought to import strange
tps files (tpsdig2) for motphometrics, to a useful
pandas dataframe and a csv file.

@author: Thomsn
"""
__author__ = 'thomas m huber'
__mail__ = 'thomas.huber@evobio.eu'

import pandas as pd
import os

os.chdir('/Users/Thomsn/Desktop/posterscent/morphom')


def tps_my_panda(i_file):
    i = 0
    indic = []
    types = []
    with open(i_file) as fly:
        soup = fly.readlines()
        all_co = pd.DataFrame()
        lmolo = False
        curvolo = False
        curv_count = 0
        curv_points = []
        lm_points = []
        for pasta in soup:
            if 'LM=' in pasta:
                lm_count = int(pasta.split('\n')[0].split('LM=')[1])
                lm_points = []
                lmolo = True
            if lmolo:
                if 'POINTS=' in pasta: # probably not neccessary
                    lm_points.append(int(pasta.split('\n')[0].split('POINTS=')[1]))
                    lmolo = False
            if 'CURVES=' in pasta:
                curv_count = int(pasta.split('\n')[0].split('CURVES=')[1])
                curv_points = []
                curvolo = True
            if curvolo:
                if 'POINTS=' in pasta:
                    curv_points.append(int(pasta.split('\n')[0].split('POINTS=')[1]))
                    curvolo = False
            if '=' not in pasta:
                coord = pasta.split('\n')[0].split(' ')
                bco = pd.Series({'x': coord[0], 'y': coord[1]})
                all_co[all_co.shape[1]] = bco
                i += 1
            if i >= 1:
                if 'IMAGE=' in pasta:
                    img_name = pasta.split('\n')[0].split('IMAGE=')[1]
                    indx = [img_name]*i
                    indic.append(indx)
                    types.append(['lm']*lm_count)
                    typos = [[f'curv{curv + 1}'] * cpo for curv, cpo in zip(range(curv_count), curv_points)]
                    typos = [e for subl in typos for e in subl]
                    types.append(typos)
                    i = 0

        all_co = all_co.transpose()
        all_co['image'] = [e for subl in indic for e in subl]
        all_co['types'] = [e for subl in types for e in subl]
        all_co.to_csv(f'{i_file.split(".tps")[0]}.csv')
    return all_co

# Convert your TPS to csv! constructed for curves, so please proof its working if needed for LM!
jagga = tps_my_panda('nectars.tps')

