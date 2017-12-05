#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
This script generates code which implements DBMF lumping for a given model.
The generated script (placed in the ./output directory by default) runs independently of the toolset.
The autorun flag allows it to call the generated code directly after creation.
If the number of bins (i.e. bin_num) is set to auto, the code is generated and executed until the stopping criterion (in evaluation.py) is fulfilled.
Additional output is written into LumpingLog.log (see utilities.py for logging options).

Caution:
The code uses eval/exec, please use with sanitized input only.
Existing files are overwritten without warning.

Example usage and arguments:
python DBMF.py model/SIR.model     		# to generate a script of SIR.model

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
from DegreeClusterEngine import *
import sys

#------------------------------------------------------
# Config
#------------------------------------------------------

DELETE_UNUSED = True
SYMPY_SIMPLIFICATION = True
DEFAULT_INPUT = 'model/SIIIR.model'
RUN_DIRECTLY = False
sys.dont_write_bytecode = True


#------------------------------------------------------
# Load Model
#------------------------------------------------------

def solve_model(model, call_after_creation = RUN_DIRECTLY):
	global DELETE_UNUSED
	number_of_states = model["number_of_states"]
	degree_distribution = model["degree_distribution"]
	states = model["states"]
	initial_distribution = model["initial_distribution"]
	k_max = model["k_max"]
	independent_rules = model["independent_rules"]
	contact_rules = model["contact_rules"]
	h = model["horizon"]
	bin_num = model["bin_num"]
	model_name = model['name']
	model['output_path'] = model['output_dir']+'dbmf_'+model_name+".py"


	#------------------------------------------------------
	# Binning
	#------------------------------------------------------

	binning_method = model['heuristic']
	bins = call_bin_heuristic(model)
	model['bins'] = binning_to_str(bins)
	x = binning_to_str(bins)
	x = prune(x)
	logger.info('binning done, bins are: {}'.format(prune(binning_to_str(bins))))

	# compute bin mean, i.e. <k>_b
	bin_distribution = {b: 0.0 for b in bins}
	for i in range(len(degree_distribution)):
		bin_distribution[bins[i]] += degree_distribution[i]
	model['bin_distribution'] = bin_distribution
	model['odes_per_state'] = len(bin_distribution)
	degree_statistics = dict()
	degree_statistics['<k>'] = bin_statistics(lambda x: x, bins, degree_distribution)
	model['degree_statistics'] = degree_statistics
	mean_bin_degree = degree_statistics['<k>']

	model['number_of_odes'] = len(states) * model['odes_per_state']
	# model['binning_error'] = binning_error(bins, degree_distribution) only for evaluation

	# normalize initial vector if full init vector is specified, consider README before doing this
	if 'initial_vector' in model:
		DELETE_UNUSED = False
		model['binned_initial_vector'] = bin_initial_vector_dbmf(model)


	#------------------------------------------------------
	# Setup ODE Generation
	#------------------------------------------------------

	# contains the ODE expression for each state as a string (which is essentially python syntax)
	# the degree is indicated by a #
	pop_lines = dict()
	assert(states == sorted(states))

	for state in states:
		pop_lines[state] = 'dt_{state}[#]='.format(state=state)

	# add independent rules to expression
	for rule in independent_rules:
		disappears,appears,rate = rule
		assert(disappears != appears)
		change = '({rate}*{disappears}[#])'.format(rate = to_str(rate), disappears = disappears)
		pop_lines[disappears] += "-" + change
		pop_lines[appears] += "+" + change

	# add contact rules to expression
	for rule in contact_rules:
		disappear_pair,appear_pair,rate = rule
		assert(disappear_pair[0] != appear_pair[0] or disappear_pair[1] != appear_pair[1])

		# upper part of the rule
		if disappear_pair[0] != appear_pair[0]:
			change = '({rate}*{disappear_0}[#]*mean_bin_degree[#]*prob_nbr_{disappear_1})'.format(rate = to_str(rate), disappear_0 = disappear_pair[0], disappear_1 = disappear_pair[1])
			pop_lines[disappear_pair[0]] += "-" + change
			pop_lines[appear_pair[0]] += "+" + change

		# lower part of the rule
		if disappear_pair[1] != appear_pair[1]:
			change = '({rate}*{disappear_1}[#]*mean_bin_degree[#]*prob_nbr_{disappear_pair_0})'.format(rate = to_str(rate), disappear_0 = disappear_pair[0], disappear_1 = disappear_pair[1])
			pop_lines[disappear_pair[1]] += "-" + change
			pop_lines[appear_pair[1]] += "+" + change


	#------------------------------------------------------
	# Create Correlation Probabilities p_k[s']
	#------------------------------------------------------

	nbr_probs = dict()
	for state in states:
		line = 'prob_nbr_{state}=1.0/mean_degree*np.sum([bin_distribution[k]*{state}[k]*mean_bin_degree[k] for k in range({bin_num})])'.format(state = state, bin_num = bin_num)
		nbr_probs[state] = line


	#------------------------------------------------------
	# Remove Unused Variables
	#------------------------------------------------------

	if DELETE_UNUSED:
		# use sympy expr in future to avoid this
		seperators = " ,;,+,-,[,],=,),(,dt_,*".split(',')
		def split_line(l):
			for sep in seperators:
				l = l.replace(sep,",")
			return l.split(',')

		used_nbrs = set()
		for key, value in pop_lines.items():
			value = [v for v in split_line(value) if 'prob_nbr_' in v]
			for v in value:
				used_nbrs.add(v)

		# remove variables if the do not occur in any ODE
		nbr_probs = {k:v for k,v in nbr_probs.items() if 'prob_nbr_'+k in used_nbrs}


	#------------------------------------------------------
	# Prepare document creation
	#------------------------------------------------------

	lines = ""

	offset_dict = dict()
	def set_offset(s):
		seperators = ' ,;,+,-,=,),(,*,/,dt_'.split(',')
		for ch in seperators:
			s = s.replace(ch, ' '+ch+' ')
		for key, value in offset_dict.items():
			s = s.replace(' '+key, ' '+value)
		for ch in seperators:
			s = s.replace(' '+ch+' ', ch)
		return s

	# create header
	for i in range(len(states)):
		lines += '#{states_i}=x[{min_i}:{max_i}]\n'.format(states_i = states[i], min_i = i*bin_num, max_i = (i+1)*bin_num)
		offset_dict[states[i]+'[k]'] = 'x[k+{min_i}]'.format(min_i = i*bin_num)
		offset_dict['dt_'+states[i]+'[k]'] = 'dt_x[k+{min_i}]'.format(min_i = i*bin_num)

	lines += '\n'
	lines += 'dt_x = np.zeros(len(x), dtype=np.double)\n\n'
	for state in states:
		lines += '#dt_STATE = [0.0] * len({state})\n'.format(state = state)

	# neighbor Probabilities
	lines += '\n'
	for line in list(nbr_probs.values()):
		line = set_offset(line)
		lines += line + '\n'

	# ODEs
	lines += '\n'
	for state, code in pop_lines.items():
		line1 = "for k in range({bin_num}):".format(bin_num = bin_num)
		line2 = "\tdt_{state}[#]=".format(state = state) + code.split("=")[1]
		line2 = line2.replace('#','k')
		if SYMPY_SIMPLIFICATION:
			line2 = ode_simplify(line2)
		line2 = set_offset(line2)
		lines += (line1+'\n'+line2+'\n')


	# suffix
	suffix = "return dt_x"
	lines += "\n"+suffix

	init = ''
	for i in range(len(states)):
		init += '([{init}] * {bin_num})+'.format(init = to_str(initial_distribution[i]), bin_num = bin_num)
	init = init[:-1]
	model['init'] = init
	if 'binned_initial_vector' in model:
		model['init'] = model['binned_initial_vector']

	generate_file_dbmf(lines.split('\n'), states, k_max, h, bin_num, bin_distribution, mean_bin_degree, bins, model, init)

	# return file name depends on utilites
	logger.info('write file '+model['output_path'])

	if call_after_creation:
		return run_model(model, False)

	return model


#------------------------------------------------------
# Run model after generation
#------------------------------------------------------

def run_model(model, suppress_output = False):
	#uses exec to run generated code, stores results in model dict
	import time
	logger.info('run ODE solver')
	import sys, os
	model['output_path'] = model['output_path'].replace('\\','/')
	outfolder = model['output_path'].split('/')[:-1]
	outfolder = '/' + '/'.join(outfolder)
	outfile = model['output_path'].split('/')[-1]
	sys.path.insert(0, os.getcwd()+outfolder)
	sys.path.insert(0, model['output_dir'])

	#fix for some troubles on mac os
	time.sleep(2.5)

	exec('from ' + outfile.replace('.py','') + ' import *', globals())
	if suppress_output:
		trajectories, time, time_elapsed, steps = compute_results()
	else:
		trajectories, time, time_elapsed, steps = plot_and_write(outfolder[1:]+'/')
	model['trajectories'] = trajectories
	model['time'] = time
	model['time_elapsed'] = time_elapsed
	model['solver_steps'] = steps
	logger.info('ODE solver finished')
	return model


#------------------------------------------------------
# Main
#------------------------------------------------------

def main(inputfile, call_after_creation = RUN_DIRECTLY, unbinned=False):
	logger.info('start DBMF with '+inputfile)
	# model file -> dict
	model = model_parser.parse(inputfile, method='DBMF')
	if unbinned:
		model['bin_num'] = model['maximal_degree_clusters']
	if model['bin_num'] == -1 and not unbinned:
		#determine number of bins heuristically
		import stopping_heuristic
		RUN_DIRECTLY = False
		stopping_heuristic.solve_model_autobin(model, method='DBMF')
	else:
		solve_model(model, call_after_creation)


#------------------------------------------------------
# Example usage
#------------------------------------------------------

if __name__ == '__main__':
	#determine input model and autorun check
	# if len(sys.argv) <= 1:
	# 	inputfile = DEFAULT_INPUT
	# else:
	# 	inputfile = str(sys.argv[1])
	# call_after_creation = True if 'autorun' in str(sys.argv).lower() else RUN_DIRECTLY
	# unbinned = 'unbinned' in str(sys.argv).lower()
	# main(inputfile, call_after_creation, unbinned)

	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('model',  help="path to modelfile")
	parser.add_argument('--noautorun', action='store_true', help="generate code without executing it")
	parser.add_argument('--nolumping', action='store_true', help="generate original equations without lumping")
	args = parser.parse_args()
	main(args.model, not args.noautorun, args.nolumping)
