#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
This script generates code which implements AME lumping for a given model.
The generated script (placed in the ./output directory by default) runs independently of the toolset.
The autorun flag allows it to call the generated code directly after creation.
If the number of clusters (i.e. bin_num) is set to auto, the code is generated and executed until the stopping criterion (in evaluation.py) is fulfilled.
Additional output is written into LumpingLog.log (see utilities.py for logging options).

Caution:
The code uses eval/exec, please use with sanitized input only.
Existing files are overwritten without warning.

Example usage and arguments:
python ame.py model/SIR.model     		# to generate a script of SIR.model

See the README.md for more optinos.

For more information we refer to:
Kyriakopoulos et al. "Lumping of Degree Based Mean Field and Pair Approximation Equations for Multi State Contact Processes"

Website:
https://mosi.uni-saarland.de/?page_id=lumpy

Tested with Python 3.5.2.
"""

__author__ = "Gerrit Grossmann"
__copyright__ = "Copyright 2016, Gerrit Grossmann, Group of Modeling and Simulation at Saarland University"
__license__ = "GNU GPLv3"
__version__ = "0.1"
__email__ = "gerrit.grossmann@uni-saarland.de"

#------------------------------------------------------
# Code Starts Here
#------------------------------------------------------

import model_parser
import numpy as np
from utilities import *
import sys
import sympy
import time
import pandas as pd
import os
from LumpEngine import lump
from ClusterEngine import clustering, plot_clustering
from ExprGenerator import gen_ame, gen_beta
sys.dont_write_bytecode = True
import scipy
import matplotlib
matplotlib.use('agg')	#run without an X-server
from stopping_heuristic import *


#------------------------------------------------------
# Generate ODE expressions
#------------------------------------------------------

def generate_line(line_def_tuple):
	# helper function for generate_odes, generates a single equation
	s, m, model = line_def_tuple
	k = np.sum(m)
	line = gen_ame(s,k,m,model['independent_rules'],model['contact_rules'],model['states'])
	return [line, m, k, s, model['degree_distribution'][k]]

def generate_odes(model):
	# generates equations and beta-expressions
	import itertools
	from multiprocessing import Pool, cpu_count

	states = model['states']
	pool = Pool(cpu_count())
	state_combinatinos = list(itertools.product(states,states,states))
	beta_exprs = pool.map(gen_beta, state_combinatinos)
	pool.close()
	pool.join()
	
	pool = Pool(cpu_count())
	degrees = range(model['k_max']+1)
	m_vecs = list()
	for k in degrees:
		m_vecs +=  sorted(list(m_k_of(k, len(states))))
	formula_combinations = list(itertools.product(states,m_vecs, [model]))
	odes = pool.map(generate_line, formula_combinations)
	pool.close()
	pool.join()
	return odes, beta_exprs

def generate_odes_old(model):
	beta_exprs = list()
	odes = list()
	states = model['states']
	for s in states:
		for s1 in states:
			for s2 in states:
				line = gen_beta(s, s1, s2)
				beta_exprs += [line]

	for s in states:
		for k in range(model['k_max']+1):
			for m in sorted(list(m_k_of(k, len(states)))):
				line = gen_ame(s,k,m,model['independent_rules'],model['contact_rules'],states)
				odes.append([line, m, k, s, model['degree_distribution'][k]])
	return odes, beta_exprs
#------------------------------------------------------
# Convert to pandas dataframe
#------------------------------------------------------

def to_dataframe(odes):
	# genereate dataframe from line, probably unnecessary in future
	# should probably be done directly in generate_odes
	labels = 'ode,neighborhood,degree,state,weight'.split(',')
	ode_frame = pd.DataFrame.from_records(odes, columns=labels)
	return ode_frame

#------------------------------------------------------
# Normalize initial values
#------------------------------------------------------

# All values corresponding to one particular degree should add up to one.
# This block normalizes the fractions/probabilities accordingly.  
# Either a multinomial distribution for a uniform distribution is used 
# for different neighborhood vectors. 

def normalize_init(ode_frame, model):
	init_dist = create_normalized_np(model['initial_distribution']) 
	states = model['states']
	if 'init' in model and model['init'].lower().strip() == 'uniform':
		logger.info('use uniform init (degree-wise)')
	else:
		logger.info('use multinomial init (degree-wise)')
	def get_init_prob(row):
		m = row[0]
		s = row[1]
		density = multinomial_pmf(m,init_dist) if np.sum(m) > 0 else 1.0
		state_scale = init_dist[model['states'].index(s)]
		if 'init' in model and model['init'].lower().strip() == 'uniform':
			result =  state_scale
		else:
			result = density * state_scale
		return result
	ode_frame['init_raw'] = ode_frame[['neighborhood', 'state']].apply(get_init_prob , axis=1)

	model['max_cluster'] = len(set(ode_frame['neighborhood'].tolist()))
	logger.info('Number of equations per state is: '+str(model['max_cluster']))

	degree_count = {k: 0 for k in range(model['k_max']+1)}
	sum_dict = {k: 0.0 for k in range(model['k_max']+1)}
	for _, row in ode_frame.iterrows():
		sum_dict[row['degree']] += row['init_raw']
		degree_count[row['degree']] += 1
	ode_frame['initial_value'] = ode_frame.apply(lambda row: 0 if sum_dict[row['degree']] == 0 else row['init_raw']/sum_dict[row['degree']] , axis=1)
	del ode_frame['init_raw']
	ode_frame['degree_count'] = ode_frame.apply(lambda row: degree_count[row['degree']] , axis=1)
	ode_frame.to_csv(model['output_dir']+'ame_frame_original_{}.csv'.format(model['name']), header='sep=,')

#------------------------------------------------------
# Cluster ODEs
#------------------------------------------------------

# Next, we apply the lumping from LumpEngine.py
# We fist define the clusters, during the lumping we substitute (and scale) variables
# After that we need to compute some characteristic values for each cluster, to make the
# aggregated equations mathematically sound (e.g. mixed_mom_matrix).

def apply_lumping(ode_frame, model):
	if model['bin_num'] == -1:
		model['bin_num'] = np.sum([elemsin_k_vec_with_sum_m(len(model['states']),k) for k in range(model['k_max']+1)]) #TODO unused?
	cluster_dict = clustering(model)
	model['actual_cluster_number'] = len(set(cluster_dict.values()))
	logger.info('Actual cluster number is: '+str(model['actual_cluster_number']))
	plot_clustering(cluster_dict, model['output_path'][:-3]+'_clustering.pdf')
	ode_frame['cluster_indicator'] = ode_frame.apply(lambda row: '{}_#_{}'.format(cluster_dict[row['neighborhood']],row['state']), axis=1)
	logger.info('Start lumping.')
	if 'scale_during_substitution' in model and 'equal_weight' in model:
		ode_lumpy = lump(ode_frame, scale_during_substitution = eval(model['scale_during_substitution']), equal_weight = eval(model['equal_weight']))
	elif 'scale_during_substitution' in model:
		ode_lumpy = lump(ode_frame, scale_during_substitution = eval(model['scale_during_substitution']))
	elif 'equal_weight' in model:
		ode_lumpy = lump(ode_frame, equal_weight = eval(model['equal_weight']))
	else:
		ode_lumpy = lump(ode_frame)
	logger.info('Lumping done.')

	def agg_mvec(line): #TODO  make this part of lumping with sympy
		weight = line['weight_normalized']
		mvecs = line['neighborhood']
		assert(len(weight) == len(mvecs))
		base = np.zeros(len(mvecs[0]))
		for i in range(len(mvecs)):
			m = np.array(mvecs[i])*weight[i]    *line['degree_count_avg']/line['degree_count'][i]
			base += m
		return tuple(base)

	def agg_mixedmom(line):
		weight = line['weight_normalized']
		mvecs = line['neighborhood']
		assert(len(weight) == len(mvecs))
		base = np.zeros([len(mvecs[0]),len(mvecs[0])])
		for i in range(len(mvecs)):
			m = mvecs[i]
			w = weight[i]
			scale = line['degree_count_avg']/line['degree_count'][i]
			for i1, s1 in enumerate(m):
				for i2, s2 in enumerate(m):
					v = m[i1]*m[i2]*scale *w
					base[i1,i2] += v
		return repr(base).replace('array(','').replace(')','')

	ode_lumpy['m'] = ode_lumpy.apply(agg_mvec, axis=1)
	ode_lumpy['mixed_mom_matrix'] = ode_lumpy.apply(agg_mixedmom, axis=1)
	ode_lumpy['name'] = ode_lumpy.apply(lambda l: l['ode'].split('=')[0].replace('dt_x["','').replace('"]',''), axis=1)
	ode_lumpy = ode_lumpy.rename(columns={'weight_sum':'degree_prob_sum'})
	ode_lumpy['state'] = ode_lumpy.apply(lambda l: l['state'][0], axis=1)
	return ode_lumpy


#------------------------------------------------------
# Write Data
#------------------------------------------------------

def write_data(ode_lumpy, beta_exprs, model):
	ode_lumpy.to_csv(model['output_dir']+'ame_frame_lumped_{}.csv'.format(model['name']), header='sep=,')
	ode_str = ''
	for line in beta_exprs:
		ode_str += '\t' + str(line) + '\n'
	for _, ode in ode_lumpy.iterrows():
		ode_str += '\t{ode}\n'.format(ode=ode['ode'])
	model['ode_text'] = ode_str
	return genrate_file_ame(model)


#------------------------------------------------------
# Solve ODE
#------------------------------------------------------

def solve_ode(model):
	from time import sleep
	logger.info('Start ODE solver.')
	folderpath = model['output_dir']
	filename = model['output_name']
	sys.path.append(folderpath)
	exec('import {} as odecode'.format(filename[:-3]), globals())
	results, t, time_elapsed = odecode.plot()
	model['trajectories'] = results
	model['time'] = t
	model['time_elapsed'] = time_elapsed
	logger.info('ODE solver done.')

#------------------------------------------------------
# Main
#------------------------------------------------------

def generate_and_solve(model, autorun, unbinned):
	model_parser.set_modelpaths(model, overwrite_dir=False) # to make paths consistent
	if unbinned:
		model['bin_num'] = -1
	logger.info('Generate ODEs.')
	odes, beta_exprs = generate_odes(model)
	logger.info('Generate ODEs finished.')
	ode_frame = to_dataframe(odes)
	model['neighborhood'] = list(set(ode_frame['neighborhood']))
	normalize_init(ode_frame, model)
	ode_lumpy = apply_lumping(ode_frame, model)
	logger.info('Write File.')
	outpath = write_data(ode_lumpy, beta_exprs, model)
	logger.info('Filepath:\t'+outpath)
	if autorun:
		sol = solve_ode(model)
		model['sol'] = sol
	logger.info('Done.')
	return model

def main(modelpath, autorun, unbinned):
	model = read_model(modelpath)
	if model['bin_num'] == -1 and not unbinned:
		return stopping_heuristic(modelpath)
	return generate_and_solve(model, autorun, unbinned)

if __name__ == '__main__':

	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('model',  help="path to modelfile")
	parser.add_argument('--noautorun', action='store_true', help="generate code without executing it")
	parser.add_argument('--nolumping', action='store_true', help="generate original equations without lumping")
	args = parser.parse_args()
	main(args.model, not args.noautorun, args.nolumping)
