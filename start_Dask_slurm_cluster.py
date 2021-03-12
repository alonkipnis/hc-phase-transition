from dask_jobqueue import SLURMCluster
import os

def start_Dask_cluster() :
	return cluster = SLURMCluster(n_workers=1, # use cluster.scale(n_workers) later on
		queue = 'owners',  # partition (-p). Other options are 'donoho', 'hns'
		walltime='0:30:00', # When workers reach their walltime they are restarte
		cores=32,	   # total number of cores
		memory="4GB",	   # not sure if per worker or in total
		local_directory=os.environ['SCRATCH']  # optional
		)


