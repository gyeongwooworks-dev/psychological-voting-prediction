from itertools import combinations
from string import ascii_lowercase

import numpy as np
import pandas as pd

QUESTIONS = list(ascii_lowercase[:20])
ANSWERS   = [f'Q{q}A' for q in QUESTIONS]
DELAYS    = [f'Q{q}E' for q in QUESTIONS]
TPS       = [f'tp{str(i).zfill(2)}' for i in range(1, 11)]
NEEDENCO  = ['age_group', 'gender', 'race', 'religion']
TARGET    = 'voted'

NEG_ITEMS_MACH = ['QeA', 'QfA', 'QkA', 'QqA', 'QrA']
NEG_ITEMS_BIG5 = ['QaA', 'QdA', 'QgA', 'QiA', 'QnA']

FORWARD = ['QbA', 'QcA', 'QhA', 'QjA', 'QmA', 'QoA', 'QsA']
REVERSE = ['QeA', 'QfA', 'QkA', 'QqA', 'QrA']
SECRET  = ['QaA', 'QdA', 'QgA', 'QiA', 'QlA', 'QnA', 'QpA', 'QtA']


def build_features_m1(df_list):
    """Model 1 피처 엔지니어링 (VotingClassifier용)"""
    for data in df_list:
        data['T'] = data['QcA'] - data['QfA'] + data['QoA'] - data['QrA'] + data['QsA']
        data['V'] = data['QbA'] - data['QeA'] + data['QhA'] + data['QjA'] + data['QmA'] - data['QqA']
        data['M'] = -data['QkA']

        for flip in NEG_ITEMS_MACH:
            data[flip] = 6 - data[flip]
        for flip in NEG_ITEMS_BIG5:
            data[flip] = 6 - data[flip]

        data['Mach_score'] = data[ANSWERS].mean(axis=1)
        data['delay']      = data[DELAYS].sum(axis=1) ** (1 / 10)

        for a, b in combinations(ANSWERS, 2):
            data[f'{a}_dv_{b}'] = data[a] / data[b].replace(0, 1e-9)

        data.drop(ANSWERS + DELAYS, axis=1, inplace=True)
        data.drop('hand', axis=1, inplace=True)

        wr_list = [f'wr_0{i}' for i in range(1, 10)] + [f'wr_{i}' for i in range(10, 14)]
        wr_keep = ['wr_01', 'wr_03', 'wr_06', 'wr_09', 'wr_11']
        data.drop([c for c in wr_list if c not in wr_keep], axis=1, inplace=True)

        data['Ex']  = data['tp01'] - data['tp06']
        data['Ag']  = data['tp07'] - data['tp02']
        data['Con'] = data['tp03'] - data['tp08']
        data['Es']  = data['tp09'] - data['tp04']
        data['Op']  = data['tp05'] - data['tp10']
        data.drop([f'tp0{i}' for i in range(1, 10)] + ['tp10'], axis=1, inplace=True)


def build_features_m2(df_list):
    """Model 2 피처 엔지니어링 (LGBM Optuna + RFE용, 강화버전)"""
    for data in df_list:
        data['T'] = data['QcA'] - data['QfA'] + data['QoA'] - data['QrA'] + data['QsA']
        data['V'] = data['QbA'] - data['QeA'] + data['QhA'] + data['QjA'] + data['QmA'] - data['QqA']
        data['M'] = -data['QkA']

        for flip in NEG_ITEMS_MACH:
            data[flip] = 6 - data[flip]
        for flip in NEG_ITEMS_BIG5:
            data[flip] = 6 - data[flip]

        data['Mach_score'] = data[ANSWERS].mean(axis=1)
        data['mach_var']   = data[ANSWERS].var(axis=1)
        data['mach_std']   = data[ANSWERS].std(axis=1)
        data['mach_min']   = data[ANSWERS].min(axis=1)
        data['mach_max']   = data[ANSWERS].max(axis=1)
        data['mach_range'] = data['mach_max'] - data['mach_min']

        data['delay']      = data[DELAYS].sum(axis=1) ** (1 / 10)
        data['delay_log']  = np.log1p(data['delay'])
        data['delay_max']  = data[DELAYS].max(axis=1)
        data['delay_std']  = data[DELAYS].std(axis=1)
        data['delay_skew'] = data[DELAYS].skew(axis=1)

        for a, b in combinations(ANSWERS, 2):
            data[f'mach_{a}_dv_{b}'] = data[a] / data[b].replace(0, 1e-9)

        for tp in TPS:
            data[tp] = 7 - data[tp]
            data[tp] = data[tp].replace(0, np.nan).fillna(data[tp].mean())

        data['Ex']  = data['tp01'] - data['tp06']
        data['Ag']  = data['tp07'] - data['tp02']
        data['Con'] = data['tp03'] - data['tp08']
        data['Es']  = data['tp09'] - data['tp04']
        data['Op']  = data['tp05'] - data['tp10']

        big5 = ['Ex', 'Ag', 'Con', 'Es', 'Op']
        data['big5_sum']  = data[big5].sum(axis=1)
        data['big5_mean'] = data[big5].mean(axis=1)
        for a, b in combinations(big5, 2):
            data[f'b5_{a}_dv_{b}'] = data[a] / (data[b].replace(0, 1e-9))

        for a, b in combinations(TPS, 2):
            data[f'tp_{a}_dv_{b}'] = data[a] / data[b].replace(0, 1e-9)

        data['teenager_ox'] = (data['age_group'] == '10s').astype(int)

        age_map = {'10s': 1, '20s': 2, '30s': 3, '40s': 4, '50s': 5, '60s': 6}
        data['age_num']    = data['age_group'].map(age_map).fillna(3)
        data['mach_x_age'] = data['Mach_score'] * data['age_num']
        data['mach_x_Es']  = data['Mach_score'] * data['Es']
        data['mach_x_Con'] = data['Mach_score'] * data['Con']
