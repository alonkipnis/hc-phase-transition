import numpy as np
from TwoSampleHC import two_sample_pvals, HC
import pandas as pd
import yaml

import logging
logging.basicConfig(level=logging.INFO)

import dask
import dask.dataframe as dd
from dask.distributed import Client

import argparse


def gen_params(len_r=2, len_beta=2, nMonte=1, N=100, lo_n=[1e3], lo_xi=[0]) :
    """
    Generating experiment parameters
    
    """
    
    rr=np.concatenate([np.array([0.0]), np.linspace(0.1, 3, len_r)]) 
    bb=np.linspace(0.45, 0.99, len_beta)
    
    N = 1e4
    nn = lo_n
    ee = np.round(N ** (-bb),6)
    mm = np.round(np.sqrt(2*np.log(N) * rr),3)
    xx = lo_xi
    for itr in range(nMonte) :
        for n in nn :
            for eps in ee :
                for mu in mm :
                    for xi in xx :
                        yield {'itr' : itr, 'n' : n, 'N': N,
                               'ep' : eps, 'mu' : mu, 'xi' : xi} 

def sample_from_mixture(lmd0, lmd1, eps) :
    N = len(lmd0)
    idcs = np.random.rand(N) < eps
    #idcs = np.random.choice(np.arange(N), k)
    lmd = np.array(lmd0.copy())
    lmd[idcs] = np.array(lmd1)[idcs]
    return np.random.poisson(lam=lmd)

def power_law(n, xi) :
    p = np.arange(1.,n+1) ** (-xi)
    return p / p.sum()

def evaluate_iteration(n = 10, N = 10, ep = .1, mu = 1, xi = 0, metric = 'Hellinger') :
    logging.debug(f"Evaluating with: n={n}, N={N}, ep={ep}, mu={mu}, xi={xi}")
    P = power_law(N, xi)
    
    if metric == 'Hellinger' :
      QP = (np.sqrt(P) + np.sqrt(mu))**2

    if metric == 'ChiSq' :
      QP = P + 2 * np.sqrt(P * mu)

    if metric == 'proportional' :
      QP = P *( 1 + r * np.log(N))

    if metric == 'power' :
      QP = P * (np.log(N) ** r)

    smp1 = sample_from_mixture(n*P, n*QP, ep)
    smp2 = sample_from_mixture(n*P, n*QP, ep)

    min_cnt = 0
    stbl = False
    gamma = 0.25

    pv = two_sample_pvals(smp1, smp2, randomize=True, sym=True)
    pv = pv[(smp1 == 0) | (smp2 == 0)]

    if len(pv) > 0 :
        hc, _ = HC(pv[pv < 1], stbl=stbl).HC(gamma=gamma)
        MinPv = -np.log(pv.min())
    else :
        print("empty")
        hc = np.nan
        MinPv = np.nan

    pv_NR = two_sample_pvals(smp1, smp2, randomize=False)
    pv_NR = pv_NR[(smp1 == 0) | (smp2 == 0)]
    
    if len(pv_NR) > 0 :
        hc_NR, _ = HC(pv_NR[pv_NR < 1], stbl=stbl).HC(gamma=gamma)
        MinPvNR = -np.log(pv_NR.min())
    else :
        print("empty")
        hc_NR = np.nan
        MinPvNR = np.nan

    return {'HC_NR' : hc_NR, 'minPv_NR' : MinPvNR,
            'HC' : hc, 'minPv' : MinPv}


def main() :
    
    parser = argparse.ArgumentParser(description='Run two-sample HC phase transition experiment')
    parser.add_argument('-o', type=str, help='output file', default='results.csv')
    parser.add_argument('-n', type=int, help='number of Monte Carlo iterations', default=1)
    parser.add_argument('-params', type=str, help='yaml parameters file.', default='params.yaml')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    #
    if args.test :
        df = pd.DataFrame(gen_params(nMonte=args.n))
        y = df.apply(lambda row : evaluate_iteration(n=row['n'], N=row['N'],
                                                             ep=row['ep'], mu=row['mu'], 
                                                             xi=row['xi'], metric='Hellinger'
                                                            ), axis=1)

    else :
        param_file = args.params
        logging.info(f" Reading parameters from {param_file}:")
        with open(param_file) as file:
            params = yaml.load(file, Loader=yaml.FullLoader)
        
        logging.info(" Running using dask.")
        
        df = pd.DataFrame(gen_params(**params))
        ddf = dd.from_pandas(df, npartitions=4)
        
        logging.info(" Connecting to dask server...")
        client = Client()
        # compute
        
        x = ddf.apply(lambda row : evaluate_iteration(n=row['n'], N=row['N'], ep=row['ep'],
                                                mu=row['mu'], xi=row['xi'], metric='Hellinger'
                                                              ), axis=1, meta=dict)
        logging.info(" Sending futures...")
        y = x.compute()
        
    df_res = pd.json_normalize(y)
        
    logging.info(f" Saving results to {args.o}...")
    results = pd.concat([df, df_res], axis=1)
    results.to_csv(args.o)

if __name__ == '__main__':
    main()
