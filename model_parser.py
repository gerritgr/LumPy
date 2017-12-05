import re
import numpy as np
from utilities import *
#from binning import get_allowed_heuristics
import math # might be needed for some lambda evaluations


#------------------------------------------------------
# Load model
#------------------------------------------------------

def set_modelpaths(model, overwrite_dir=True):
	import os
	if overwrite_dir:
		model['output_dir'] = './output/{}/'.format(model['name'])
	model['output_path'] = model['output_dir']+'ame_{}.py'.format(model['name'])
	model['output_path'] = os.path.abspath(model['output_path'])
	model['output_dir'] = os.path.dirname(model['output_path'])+'/'
	model['output_name'] = os.path.basename(model['output_path'])
	if not os.path.exists(model['output_dir']):
		os.makedirs(model['output_dir'])
	return model


def initial_vector(distribution, states):
	assert(states == sorted(states))
	if len(states) < 2:
		raise ValueError('Number of states must be >= 2.')

	init_d = None
	if type(distribution) is list:
		init_d = [float(p) for p in distribution]
	if type(distribution) is dict:
		init_d = list()
		for s in states:
			init_d.append(float(distribution.get(s,0.000001)))
	if hasattr(distribution, '__call__'):
		init_d = list()
		for s in states:
			init_d.append(float(distribution(s)))

	if init_d is None:
		raise ValueError('Initial distribution specification is incorrect.')
	if type(distribution) is list and len([prob for prob in distribution if prob <= 0.0]) > 0:
		raise ValueError('All initial probabilites must be positive.')

	partition = np.sum(init_d)
	if isclose(partition, 0.0):
		raise ValueError('Initial distribution specification is incorrect.')
	if not isclose(1.0, partition):
		logger.info('Normalize degree distribution, as it adds up to ' + str(partition))
		init_d = [prob/partition for prob in init_d]
	if len(states) != len(init_d):
		raise ValueError('Number of states does not match initial distribution.')
	return init_d

def degree_vector(distribution, number_of_degrees):
	d = None
	if type(distribution) is list:
		d = [float(p) for p in distribution]
		if len(d) != number_of_degrees:
			raise ValueError('Number of degrees (i.e. k_max) does not match degree distribution.')
	elif type(distribution) is dict:
		d = list()
		for k in range(number_of_degrees):
			d.append(float(distribution.get(k, 0.0)))
	elif hasattr(distribution, '__call__'):
		d = list()
		for k in range(number_of_degrees):
			d.append(float(distribution(k)))

	if d is None:
		raise ValueError('Degree distribution specification is incorrect.')
	if len([prob for prob in d if prob < 0.0]) > 0:
		raise ValueError('Probabilites in degree distribution cannot be negative.')
	partition = np.sum(d)
	if isclose(partition, 0.0):
		raise ValueError('Degree distribution specification is incorrect.')
	if not isclose(1.0, partition):
		logger.warning('Normalize degree distribution, as it adds up to '+ str(partition))
		d = [prob/partition for prob in d]
	return d


def parse(filename = 'model.model', method = 'AME'):
	import ast #secure eval
	assert(method in ['AME', 'PA', 'DBMF'])

	logger.info('parse file: '+filename)

	f = open(filename, 'r')
	t = f.read()
	f.close()

	lines = t.split('\n')

	lines = [l.split("#")[0] for l in lines]
	lines = [l.replace('\t','') for l in lines if not l.isspace()]

	model = dict()

	def cl(s):
		if not type(s) is str:
			return s
		return s.replace("'","").replace('"','')

	number_of_states = None
	degree_distribution = None
	states = None
	initial_distribution = None
	k_max = None
	independent_rules = list()
	contact_rules = list()
	horizon = None
	bin_num = None
	heuristic = None
	modeltext = ''

	name = filename.replace('\\',"/").split('/')[-1].replace('.model','')

	for l in lines:
		# mo = re.search("(.*)number_of_nodes(.*)=\s*([0-9]+)\s*", l)
		# if mo is not None:
		# 	number_of_nodes = int(mo.group(3))
		# 	continue
		modeltext += '   ---   '+l

		mo = re.search("(.*)horizon(.*)=\s*(\d+\.\d+)\s*", l)
		if mo is not None:
			horizon = float(mo.group(3))
			if horizon <= 0.0:
				raise ValueError('Horizion must be positive.')
			continue
		mo = re.search("(.*)horizon(.*)=\s*([0-9]+)\s*", l)
		if mo is not None:
			horizon = float(mo.group(3))
			if horizon <= 0.0:
				raise ValueError('Horizion must be positive.')
			continue
		mo = re.search("(.*)k_max(.*)=\s*([0-9]+)\s*", l)
		if mo is not None:
			k_max = int(mo.group(3))
			# for debugging
			#k_max = int(k_max*0.75)
			#logger.error('kmax is set incorrectly.')
			if k_max < 2:
				raise ValueError('k_max is too small')
			continue
		mo = re.search("(.*)bin_num(.*)=\s*(-?[0-9]+)\s*", l)
		if mo is not None:
			bin_num = int(mo.group(3))
			if bin_num <= 0 and bin_num != -1:
				raise ValueError('bin_num must be positive or -1 for auto detection.')
			continue
		if 'bin_num' in l and '=' in l and 'auto' in l.split('=')[1].lower():
			bin_num = -1
		mo = re.search("(.*)heuristic(.*)=(.*)", l)
		if mo is not None:
			heuristic = mo.group(3).strip()
		mo = re.search("(.*)name(.*)=(.*)", l)
		mo = re.search("(.*)output_dir(.*)=(.*)", l)
		if mo is not None:
			output_dir = mo.group(3).strip()
			output_dir = output_dir.replace('\\','/')
			if not output_dir.endswith('/'):
				output_dir += '/'
		# if not output_dir.replace('/','').isalnum():
		# 	raise ValueError('Output directory must be alphanumeric.')
		mo = re.search("(.*)name(.*)=(.*)", l)
		if mo is not None:
			name = mo.group(3).strip().lower()
			if not name.isalpha():
				raise ValueError('The name can only contain letters: '+name+' is not possible.')
			continue
		mo = re.search("(.*)degree_distribution(\s*)=\s*(.*?)\s*$", l)
		if mo is not None:
			degree_distribution = eval(mo.group(3))
			continue
		mo = re.search("(.*)states\s*=\s*(.*?)\s*$", l)
		if mo is not None:
			l = mo.group(2)
			l = l.replace(";",",")
			for r in ["[","]","'",'"','(',")", " "]:
				l = l.replace(r,"")
			states = l.split(",")
			states = sorted(states)
			if len(states) <= 1:
				raise ValueError('Number of states is too low.')
			for state in states:
				if not state.isalpha():
					raise ValueError('The name of a state can only contain letter: '+state+' is not possible.')
			continue
		mo = re.search("(.*)initial_distribution\s*=\s*(.*?)\s*$", l)
		if mo is not None:
			l = mo.group(2)
			if ":" in l or 'lambda' in l:
				# is in dict syntax
				initial_distribution = eval(l)
			else:
				l = l.replace(";",",")
				for r in ["[","]","'",'"','(',")", " "]:
					l = l.replace(r,"")
				initial_distribution = l.split(",")
			continue
		mo = re.search("\s*R(.*?)\s*:\s*(.*?)\s*\+\s*(.*?)\s*->\s*(.*?)\s*\+\s*(.*?)\s*with\s*(.*?)\s*$", l)
		if mo is not None:
			c_rule = mo.group(2)
			rate = float(ast.literal_eval(mo.group(6)))
			if rate <= 0.0:
				raise ValueError("rate <= 0")
			if cl(mo.group(2)) != cl(mo.group(4)) and cl(mo.group(3)) != cl(mo.group(5)):
				raise ValueError("Rules of the form A+B -> C+D are not allowed, error in line: "+l)
			if cl(mo.group(2)) == cl(mo.group(4)) and cl(mo.group(3)) == cl(mo.group(5)):
				raise ValueError("Rules of the form A+B -> A+B are not allowed, error in line: "+l)
			if cl(mo.group(2)) != cl(mo.group(4)):
				contact_rules.append(((cl(mo.group(2)),cl(mo.group(3))),(cl(mo.group(4)),cl(mo.group(5))),rate))
			else: # make sure only the first state changes.
				contact_rules.append(((cl(mo.group(3)),cl(mo.group(2))),(cl(mo.group(5)),cl(mo.group(4))),rate))
			continue
		mo = re.search("\s*R(.*?)\s*:\s*(.*?)\s*->\s*(.*?)\s*with\s*(.*?)\s*$", l)
		if mo is not None:
			i_rule = mo.group(2)
			rate = float(ast.literal_eval(mo.group(4)))
			if rate <= 0.0:
				raise ValueError("rate <= 0.0")
			independent_rules.append((cl(mo.group(2)),cl(mo.group(3)),rate))
			continue
		mo = re.search("(.*)initial_vector\s*=(.*)", l)
		if mo is not None:
			# some additional parameters
			v = mo.group(2).strip()
			model['initial_vector'] = eval(v)
			continue
		mo = re.search("(.*)=(.*)", l)
		if mo is not None:
			# some additional parameters
			model[mo.group(1).strip()] = mo.group(2).strip()


	number_of_states = len(states)

	if len(independent_rules) + len(contact_rules) == 0:
		raise ValueError('no rules specified')

	# note that he normalization of initial_vector happens during the binning,
	# as it is not cleare yet if PA or DBMF is used
	if initial_distribution is None and 'initial_vector' not in model:
		raise ValueError('Initial distribution not specified')
	if 'initial_vector' in model and 'list' not in str(type(model['initial_vector'])):
		raise ValueError('Initial vector is not a list.')
	if 'initial_vector' in model:
		if 0 != len([v for v in model['initial_vector'] if v<0.0]):
			raise ValueError('Initial vector is ill-formed.')
		logger.info('Caution when using initial_vector option, lengh and strcuture must match auto-generatred init-state.')

	degree_distribution = degree_vector(degree_distribution, k_max+1)
	initial_distribution = initial_vector(initial_distribution, states)

	for rule in independent_rules:
		if rule[0] not in states or rule[1] not in states:
			raise ValueError("Rule uses unspecified states: "+str(rule))
	for rule in contact_rules:
		if rule[0][0] not in states or rule[0][1] not in states or rule[1][0] not in states or rule[1][1] not in states:
			raise ValueError("Rule uses unspecified states: "+str(rule))
	# if not bin_num <= len([prob for prob in degree_distribution if not isclose(prob, 0.0)]):
	# 	raise ValueError('Number of bins must be smaller/equal number of degrees with positive probability!')

	assert(states == sorted(states))


	model['maximal_degree_clusters'] = len([prob for prob in degree_distribution if not isclose(prob, 0.0)])

	model["number_of_states"] = number_of_states
	model["degree_distribution"] = degree_distribution
	model["states"] = states
	model["initial_distribution"] = initial_distribution
	model["k_max"] = k_max
	model["independent_rules"] = independent_rules
	model["contact_rules"] = contact_rules
	model["horizon"] = horizon
	model["bin_num"] = int(bin_num)
	if heuristic is None and method == 'AME':
		model["heuristic"] = 'cluster_subspaceXY'
	elif heuristic is None and method in ['PA', 'DBMF']:
		model["heuristic"] = 'hierarchical2d'
	model['name'] = name
	model['modeltext'] = modeltext

	logger.info("__ parsed model __")
	logger.info(model_to_str(model))
	logger.info('__ done parsing __')

	if 'eval_points' in model:
		model['eval_points'] = int(model['eval_points'])
	else:
		model['eval_points'] = 1001

	for key, value in model.items():
		if value is None:
			raise ValueError(key+" is not specified in model.")

	set_modelpaths(model)
	return model
