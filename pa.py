#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
This script generates code which implements PA lumping for a given model.
The generated script (placed in the ./output directory by default) runs independently of the toolset.
The autorun flag allows it to call the generated code directly after creation.
If the number of bins (i.e. bin_num) is set to auto, the code is generated and executed until the stopping criterion (in evaluation.py) is fulfilled.
Additional output is written into LumpingLog.log (see utilities.py for logging options).

Caution:
The code uses eval/exec, please use with sanitized input only.
Existing files are overwritten without warning.

Example usage and arguments:
python pa.py model/SIR.model     		# to generate a script of SIR.model

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

DELETE_UNUSED = False
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
	model['output_path'] = model['output_dir']+'pa_'+model_name+".py"


	#------------------------------------------------------
	# Binning
	#------------------------------------------------------

	binning_method = model['heuristic']
	bins = call_bin_heuristic(model)
	model['bins'] = binning_to_str(bins)
	logger.info('binning done, bins are: {}'.format(prune(binning_to_str(bins))))

	bin_distribution = {b: 0.0 for b in bins}
	for i in range(len(degree_distribution)):
		bin_distribution[bins[i]] += degree_distribution[i]
	model['bin_distribution'] = bin_distribution
	model['odes_per_state'] = len(bin_distribution)
	degree_statistics = dict()
	degree_statistics['<k>'] = bin_statistics(lambda x: x, bins, degree_distribution)
	degree_statistics['<k-1>'] = bin_statistics(lambda x: x-1, bins, degree_distribution)
	degree_statistics['<k*(k-1)>'] = bin_statistics(lambda x: x*(x-1), bins, degree_distribution)
	degree_statistics['<k**2>'] = bin_statistics(lambda x: x**2, bins, degree_distribution)
	model['degree_statistics'] = degree_statistics

	#import time
	#np.random.seed(model['seed']+200)
	#time.sleep(5)
	#model['initial_vector'] = list(np.random.rand(3030))
	if 'initial_vector' in model:
		# normalize initial vector if full init vector is specified, consider README before doing this
		DELETE_UNUSED = False
		model['binned_initial_vector'] = bin_initial_vector_pa(model)


	#------------------------------------------------------
	# Setup ODE Generation
	#------------------------------------------------------

	logger.info('start with ODE generation')
	assert(states == sorted(states))

	# betas
	lines = ""
	# ODEs for fractions of states
	pop_lines = dict()
	# ODEs for probabilities that random neighbor of states s is in state s'
	ngbr_lines = dict()

	for state in states:
		pop_lines[state] = "dt_{state}[b]= ".format(state = state)

	for s1 in states:
		for s2 in states:
			p_s1_s2 = 'ngbr_{s1}_{s2}'.format(s1 = s1, s2 = s2)
			ngbr_lines[p_s1_s2] = 'dt_{p_s1_s2}[b]=-dt_{s1}[b]*{p_s1_s2}[b]/{s1}[b]'.format(p_s1_s2 = p_s1_s2, s1 = s1)


	#------------------------------------------------------
	# Generate ODEs for Population Fractions
	#------------------------------------------------------

	for rule in independent_rules:
		consume, product, rate = rule
		assert(consume != product)
		change = '({rate}*{consume}[b])'.format(rate = rate, consume = consume)
		pop_lines[consume] += '-({change})'.format(change = change)
		pop_lines[product] += '+({change})'.format(change = change)

	for rule in contact_rules:
		consume1 = rule[0][0]
		consume2 = rule[0][1]
		product1 = rule[1][0]
		product2 = rule[1][1]
		rate = rule[2]
		assert(consume1 != product1 and consume2 == product2)
		change = '({rate}*{consume1}[b]*mean_bin_degree[b]*ngbr_{consume1}_{consume2}[b])'.format(rate = rate, consume1 = consume1, consume2 = consume2)
		pop_lines[consume1] += '-{change}'.format(change = change)
		pop_lines[product1] += '+{change}'.format(change=change)


	#------------------------------------------------------
	# Generate ODEs for Correlation Probabilities
	#------------------------------------------------------

	# add independent rules
	for identity, ode in ngbr_lines.items():
		_, s1, s2 = identity.split('_')
		for rule in independent_rules:
			consume, product, rate = rule
			assert(consume != product)
			if consume == s1:
				ngbr_lines[identity] += '-({rate}*ngbr_{s1}_{s2}[b])'.format(rate = rate, s1 = s1, s2 = s2)
			if product == s1:
				ngbr_lines[identity] += '+({rate}*{consume}[b]/{product}[b]*ngbr_{consume}_{s2}[b])'.format(rate = rate, consume = consume, product = product, s1 = s1, s2 = s2)


	# add contact rules
	def get_corr_list(s1):
		l = list()
		for s2 in states:
			l +=  ['ngbr_{s1}_{s2}[b]'.format(s1 = s1, s2 = s2)]
		return l

	for identity, ode in ngbr_lines.items():
		_, s1, s2 = identity.split('_')
		for rule in contact_rules:
			consume1 = rule[0][0]
			consume2 = rule[0][1]
			product1 = rule[1][0]
			product2 = rule[1][1]
			rate = to_str(rule[2])
			assert(consume1 != product1 and consume2 == product2)

			term = ''
			if consume1 == s1 and s2 == consume2:
				term = '-({rate} * (ngbr_{s1}_{s2}[b] * (1-ngbr_{s1}_{s2}[b]*(1-mean_bin_degree[b]))))'
			elif consume1 == s1 and s2 != consume2:
				term = '-({rate} * mean_km1[b]*ngbr_{s1}_{c2}[b]*ngbr_{s1}_{s2}[b])'
			elif product1 == s1 and s2 == consume2:
				term = '+({c1}[b]/{s1}[b] * {rate}*(ngbr_{c1}_{s2}[b] * (1-ngbr_{c1}_{s2}[b]*(1-mean_bin_degree[b]))))'
			elif product1 == s1 and s2 != consume2:
				term = '+({c1}[b]/{s1}[b]*{rate}*mean_km1[b]*ngbr_{c1}_{c2}[b]*ngbr_{c1}_{s2}[b])'
			else:
				continue

			term = term.format(s1 = s1, s2 = s2, c1 = consume1, c2 = consume2, rate = rate)

			ngbr_lines[identity] += term

	# add betas to ODEs
	for identity, ode in ngbr_lines.items():
		_, s1, s2 = identity.split('_')
		for state in states:
			if s2 == state:
				continue
			ngbr_lines[identity] += '+beta_{s1}_{state}_to_{s1}_{s2}*ngbr_{s1}_{state}[b]'.format(s1 = s1, s2 = s2, state = state)
		for state in states:
			if s2 == state:
				continue
			ngbr_lines[identity] += '-beta_{s1}_{s2}_to_{s1}_{state}*ngbr_{s1}_{s2}[b]'.format(s1 = s1, s2 = s2, state = state)


	#------------------------------------------------------
	# Handle betas
	#------------------------------------------------------

	# generate betas
	variables = list()
	for s1 in states:
		for s2 in states:
			for s3 in states:
				if s2 == s3:
					continue
				term = 'beta_{s1}_{s2}_to_{s1}_{s3} = (np.sum([bin_distribution[b] * {s2}[b] * ({base}) for b in range(len(bin_distribution))]))/(np.sum([bin_distribution[b] * mean_bin_degree[b] * {s2}[b] * ngbr_{s2}_{s1}[b] for b in range(len(bin_distribution))]))'
				base = '0.0'
				for rule in independent_rules:
					consume1 = rule[0]
					product1 = rule[1]
					rate = to_str(rule[2])
					if consume1 == s2 and product1 == s3:
						base += '+ ({rate} * mean_bin_degree[b] * ngbr_{s2}_{s1}[b])'.format(rate = rate, s1 = s1, s2 = s2)

				for rule in contact_rules:
					consume1 = rule[0][0]
					consume2 = rule[0][1]
					product1 = rule[1][0]
					product2 = rule[1][1]
					rate = to_str(rule[2])
					if consume1 == s2 and product1 == s3:
						if consume2 == s1:
							exp = 'mixed_mom_2nd_bin(b, ngbr_{s2}_{s1}[b])'.format(s1 = s1, s2 = s2)
						else:
							exp = 'mixed_mom_2nd_bin(b, ngbr_{s2}_{s1}[b], ngbr_{s2}_{r2}[b])'.format(s1 = s1, s2 = s2, r2 = consume2)

						base  += '+ ({rate} * ({exp}))'.format(rate = rate, exp = exp)


				if SYMPY_SIMPLIFICATION:
					base = ode_simplify(base)
				if base == '0.0':
					term = 'beta_{s1}_{s2}_to_{s1}_{s3} = 0.0'
				term = term.format(base = base, s1 = s1, s2 = s2, s3 = s3, rate = rate)
				variables.append(term)


	#------------------------------------------------------
	# Simplify ODEs
	#------------------------------------------------------

	if SYMPY_SIMPLIFICATION:
		for state, formula in pop_lines.items():
			pop_lines[state] = ode_simplify(formula)

		# replace zero betas
		for ngbr in ngbr_lines:
			formula = ngbr_lines[ngbr]

			# this is neccessary because we work with string replacements
			# to make sure we get the end of the beta variable
			# use SymPy expressions here in future to avoid this
			seperators = '*,+,-,/,(,)'.split(',')
			for sep in seperators:
				formula = formula.replace(sep, ' '+sep+' ')
			formula += ' '

			for beta in variables:
				beta_key = beta.split('=')[0].strip() + ' ' #todo plus weg
				beta_formula = beta.split('=')[1].strip()
				if beta_key in formula and beta_formula == '0.0':
					logger.debug('get formula: {}'.format(formula))
					formula = formula.replace(beta_key, '0.0')
					logger.debug('converted to formula: {}'.format(formula))
			ngbr_lines[ngbr] = formula

		for ngbr, formula in ngbr_lines.items():
			ngbr_lines[ngbr] = ode_simplify(formula)


	#------------------------------------------------------
	# Remove Unused Code
	#------------------------------------------------------

	if DELETE_UNUSED:
		seperators = ' ,;,+,-,[,],=,),(,dt_,*,/'.split(',')
		def split_line(l):
			for sep in seperators:
				l = l.replace(sep,",")
			return l.split(',')

		used_ngrb_odes = set()
		for key, value in pop_lines.items():
			value = [v for v in split_line(value) if 'ngbr_' in v]
			for v in value:
				used_ngrb_odes.add(v)

		for var in variables:
			value = [v for v in split_line(var) if 'ngbr_' in v]
			for v in value:
				used_ngrb_odes.add(v)

		for _ in range(len(ngbr_lines) * len(ngbr_lines)):
			for key, value in ngbr_lines.items():
				if key in used_ngrb_odes:
					value = [v for v in split_line(value) if 'ngbr_' in v]
					for v in value:
						used_ngrb_odes.add(v)

		used_ngrb_vars = set()
		for key, value in ngbr_lines.items():
			if key in used_ngrb_odes:
				value = [v for v in split_line(value) if 'beta_' in v]
				for v in value:
					used_ngrb_vars.add(v)

		# actually delte keys
		ngbr_lines = {k:v for k,v in ngbr_lines.items() if k in used_ngrb_odes}
		variables = [v for v in variables if v.split("=")[0].strip() in used_ngrb_vars]


	#------------------------------------------------------
	# Prepare Document Creation
	#------------------------------------------------------

	code = ''
	init = ''
	offset_dict = dict()
	slice_length = dict()

	def set_offset(s):
		seperators = " ,;,+,-,=,),(,*,/,dt_".split(',')
		for ch in seperators:
			s = s.replace(ch, ' '+ch+' ')
		for key, value in offset_dict.items():
			s = s.replace(' '+key, ' '+value)
		for ch in seperators:
			s = s.replace(' '+ch+' ', ch)
		return s

	for i in range(len(states)):
		code += '#{states_i}=x[{min_i}:{max_i}]\n'.format(min_i = i*bin_num, max_i = (i+1)*bin_num, states_i = states[i])
		init += '([{initial_distribution_i}] * {bin_num})+'.format(bin_num = bin_num, initial_distribution_i = to_str(initial_distribution[i]))
		offset_dict[states[i]+'[b]'] = 'x[b+{offset}]'.format(offset = i*bin_num)
		slice_length[states[i]] = (i+1)*(bin_num) - i*(bin_num)

	ngbr_keys = sorted(ngbr_lines.keys())
	for i in range(len(ngbr_keys)):
		key = ngbr_keys[i]
		code += '#{key}=x[{min_i}:{max_i}]\n'.format(key=key, min_i = (len(states))*(bin_num) + i*(bin_num), max_i = (len(states))*(bin_num) + (i+1)*(bin_num))
		state2 = key.split('_')[-1]
		state2_i = states.index(state2)
		init +=  '([{initial_distribution_state2_i}] * {bin_num})+'.format(initial_distribution_state2_i = initial_distribution[state2_i], bin_num = bin_num)
		offset_dict[key+'[b]'] = 'x[b+{offset}]'.format(offset = (len(states))*(bin_num) + i*(bin_num))
		slice_length[key] = ((len(states))*(bin_num) + (i+1)*(bin_num)) - ((len(states))*(bin_num) + i*(bin_num))

	init = init[:-1] #delte last +
	model['init'] = init
	if 'binned_initial_vector' in model:
		model['init'] = model['binned_initial_vector']
	number_of_odes = (len(states))*(bin_num) + (i+1)*(bin_num)
	model['number_of_odes'] = number_of_odes
	logger.debug('offset dict: '+str(offset_dict))

	code += '\n'

	code += 'dt_x = np.zeros(len(x), dtype=np.double)\n\n'

	ngbr_keys = sorted(ngbr_lines.keys())

	code += '\n'

	for v in variables:
		v = set_offset(v)
		code += v+"\n"
	code += "\n"
	for k, v in sorted(pop_lines.items()):
		v = set_offset(v)
		code += 'for b in range(len(bin_distribution)):\n\t{v}\n'.format(v=v)
	code += "\n"
	for k, v in sorted(ngbr_lines.items()):
		v = set_offset(v)
		code += 'for b in range(len(bin_distribution)):\n\t{v}\n'.format(v=v)
	code += '\n'

	code += 'return dt_x\n'

	generate_file_pa(code.split('\n'), model)

	logger.info('write file: {}'.format(model['output_path']))

	if call_after_creation:
		return run_model(model, False)

	return model


#------------------------------------------------------
# Run Model Code After Generation
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
	time.sleep(2.0)
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

def main(inputfile, call_after_creation = RUN_DIRECTLY, unbinned = False):
	logger.info('start PA with {}'.format(inputfile))
	model = model_parser.parse(inputfile, method='PA')
	if unbinned:
		logger.info('Set degress to '+str(model['maximal_degree_clusters']))
		model['bin_num'] = model['maximal_degree_clusters']
	#determine input model and autorun check
	if model['bin_num'] == -1 and not unbinned:
		#determine number of bins heuristically
		import stopping_heuristic
		RUN_DIRECTLY = False
		logger.info('Start stopping heuristic.')
		stopping_heuristic.solve_model_autobin(model, method='PA')
	else:
		solve_model(model, call_after_creation)


if __name__ == '__main__':
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
