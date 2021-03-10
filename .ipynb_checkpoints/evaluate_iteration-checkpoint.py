import numpy as np
import pandas as pd
import TwoSampleHC
import ray

from scipy.stats import poisson, norm, chisquare, binom

import sys
import os


from TwoSampleHC import (binom_test_two_sided_random, 
  hc_vals, two_sample_pvals, binom_test_two_sided)

def power_law(n, xi) :
    p = np.arange(1.,n+1) ** (-xi)
    return p / p.sum()

def poisson_test_random(x, lmd) :
    p_down = 1 - poisson.cdf(x, lmd)
    p_up = 1 - poisson.cdf(x, lmd) + poisson.pmf(x, lmd)
    U = np.random.rand(x.shape[0])
    prob = np.minimum(p_down + (p_up-p_down)*U, 1)
    return prob * (x != 0) + U * (x == 0)

@ray.remote     
def evaluate_iteration(a, xi, r, be, n, nMonte=10,
                 metric = 'Hellinger') :
    N = int(n ** (1/a))
    #n = int(N ** a)
    P = power_law(N, xi)
    print("r = {}, beta = {}, a = {}, xi = {}, n = {}".format(r,be,a,xi,n))
    ep = N ** (-be)
    mu = r * np.log(N) / n / 2
    
    df = pd.DataFrame()
    for iM in range(nMonte) :
        
        TH1 = np.random.rand(N) < ep/2
        TH2 = np.random.rand(N) < ep/2

        if metric == 'Hellinger' :
          QP = (np.sqrt(P) + np.sqrt(mu))**2

        if metric == 'ChiSq' :
          QP = P + 2 * np.sqrt(P * mu)

        if metric == 'proportional' :
          QP = P *( 1 + r * np.log(N))

        if metric == 'power' :
          QP = P * (np.log(N) ** r)

        Q1 = P.copy()
        Q1[TH1] = QP[TH1]
        Q1 = Q1 / Q1.sum()

        Q2 = P.copy()
        Q2[TH2] = QP[TH2]
        Q2 = Q2 / Q2.sum()

        smp1 = np.random.multinomial(n, Q1)
        smp2 = np.random.multinomial(n, Q2)
        smp_P1 = np.random.poisson(lam = n*P)

        smp_P = smp1
        smp_Q = smp2
    
        min_cnt = 0
        stbl = False
        gamma = 0.25

        pv = two_sample_pvals(smp_Q, smp_P, randomize=True, sym=True)
        #pv = pv[smp_Q + smp_P > min_cnt]
        pv[(smp_Q == 0) | (smp_P == 0)]
        hc, p_th = hc_vals(pv[pv < 1], gamma = gamma, stbl=stbl, minPv=0)

        pv_NR = two_sample_pvals(smp_Q, smp_P, randomize=False)
        hc_NR, _ = hc_vals(pv_NR[pv_NR < 1], gamma = gamma, stbl=stbl, minPv=0)

        MinPv = -np.log(pv.min())
        MinPvNR = -np.log(pv_NR.min())

        dfr = pd.DataFrame({'r': [r], 'beta' : [be], 'a' : [a], 
                            'xi' : [xi],'N' : [N], 'n' : [n],
                                'metric' : metric,
                                 'nMonte' : nMonte,
                                 'HC_NR' : hc_NR,
                                 'minPv_NR' : MinPvNR,
                                 'HC' : hc,
                                 'minPv' : MinPv,
                                })
        df = df.append(dfr, ignore_index = True)

    return df
    