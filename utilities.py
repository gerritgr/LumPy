import numpy as np
import sympy
import os
import traceback

#------------------------------------------------------
# Python version check
#------------------------------------------------------

if 1/5 == 0:
	raise RuntimeError('Do not use Python 2.x!')

#------------------------------------------------------
# Logging
#------------------------------------------------------
import logging
logger = logging.getLogger('LumpingLogger')
logger.setLevel(logging.INFO)
logpath = 'LumpingLog.log'
try:
	# Use different logger output when testing
	import run_all_tests
	logpath = 'tests/LumpingLogTest.log'
except:
	pass
fh = logging.FileHandler(logpath, mode='w') # change to a to overwrite
fh.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(process)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info('-------------------------------------------------')
logger.info('                 Start Logging                   ')
logger.info('-------------------------------------------------')

#------------------------------------------------------
# Mutliprocessing
#------------------------------------------------------

def apply_to_df(row_wise_func):
    return lambda df2: df2.apply(row_wise_func, axis=1)

def multi_apply(df, row_wise_func):
	try:
		from multiprocess import Pool, cpu_count
		import pandas as pd
		cores = cpu_count()
		data_split = np.array_split(df, cores)
		pool = Pool(cores)
		map_function = apply_to_df(row_wise_func)
		new_data = pool.map(map_function, data_split)
		data = pd.concat(new_data)
		pool.close()
		pool.join()
		del pool
		del new_data
		del data_split
		return data
	except:
		logger.error('Error during multi apply:\t'+str())
		logger.error(traceback.format_exc())
	return df.apply(row_wise_func, axis=1)


def start_process(method, args):
	from multiprocessing import Process
	global global_process_list
	if 'global_process_list' in globals():
		global_process_list = globals()['global_process_list']
	else:
		global_process_list = list()
	p = Process(target=method, args=args)
	p.start()
	global_process_list.append(p)
	return p

def join_processes():
	global global_process_list
	from multiprocessing import Process
	if 'global_process_list' not in globals():
		return
	for p in global_process_list:
		p.join()

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
	return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def float2str(f):
	if 'USE_HIGH_PRECISION_STRINGS' in globals() and not globals['USE_HIGH_PRECISION_STRINGS']:
		return str(f)
	from decimal import Decimal
	return str(Decimal(repr(f)))

def to_str(x):
	try:
		s = float2str(x)
		return s
	except:
		# try to get commas in str output
		if 'numpy' in str(type(x)):
			try:
				x = x.tolist()
			except:
				pass
		s = str(x)
		return s

def create_normalized_np(v, return_original_if_zero = False):
	v = np.array(v)
	partition = float(np.sum(v))
	if isclose(partition, 0.0):
		if return_original_if_zero:
			return np.array(v)
		raise ValueError('Cannot normalize vector, sum is zero.')
	if (v < 0.0).any():
		raise ValueError('Cannot normalize vector, negative values.')
	v = v / partition
	return v

def dict_to_liststr(d):
	'''
	Example use: dict_to_liststr({4:4.0, 2:333.777})
	'''
	if not type(d) is dict:
		return to_str(d)
	dlist = [d.get(i,0.0) for i in range(int(np.max(list(d.keys()))+1.5))]
	dliststr = [to_str(v) for v in dlist]
	return '[' +','.join(dliststr)+ ']'

def model_to_str(model):
	lines = list()
	max_len = max([len(str(key)) for key in model]) + 6
	for key, value in model.items():
		gap = ' ' * (max_len - len(str(key)))
		value_str = to_str(value)
		if len(value_str) > 5000:
			value_str = value_str[0:5000] + '...'
		lines.append('#'+str(key)+gap+value_str)
	return '\n'.join(lines)

def prune(l, cutoff = 200):
	if len(l) > cutoff:
		s = l[:cutoff]
		if 'str' in str(type(s)):
			s += '...'
		return s
	else:
		return l

def hist(l):
	histogram = dict()
	for n in l:
		v = int(n)
		if v not in histogram:
			histogram[v] = 1
		else:
			histogram[v] += 1
	return histogram

def read_model(modelfile):
	import model_parser
	return model_parser.parse(modelfile)

#------------------------------------------------------
# File generation
#------------------------------------------------------

def genrate_file_ame(model):
	ode_text = model['ode_text']
	outfolder = model['output_dir']
	import os
	if not os.path.exists(outfolder):
		os.makedirs(outfolder)

	target = open("templates/ode_python_ame.template", 'r')
	template = target.read()
	target.close()

	# execute python code in curly brackets
	template = template.replace(r'}',r'}{')
	template = template.split(r'{')
	for i in range(len(template)):
		if template[i].endswith(r'}'):
			template[i] = eval(template[i][:-1])
			template[i] = to_str(template[i])
	template = ''.join(template)

	target = open(model['output_path'], 'w')
	target.write(template)
	target.close()
	logger.info('Successfully created file.')
	return os.path.abspath(model['output_path'])



def generate_file_dbmf(lines, states, k_max, horizon, odes_per_state, bin_distribution, mean_bin_degree, binning, model, init):

	outfolder = model['output_dir']
	import os
	if not os.path.exists(outfolder):
		os.makedirs(outfolder)

	ode_text = ""

	for line in lines:
		if line.endswith("=") or line.endswith("= "):
			line += "0"
		ode_text += "\t"+line+"\n"

	colors = ['b','g','r','c','m','y','k']
	template_summary = ""
	summary = "{state}summary = [0.0] * points\nfor i in range(points):\n\tfor j in range(odes_per_state):\n\t\tode_index = j + {stateshift}\n\t\t{state}summary[i] += sol[i,ode_index]*bin_distribution[j]\n"
	for state_i in range(len(states)):
		state = states[state_i]
		stateshift = state_i * odes_per_state
		#template_summary += summary.replace("_STATE_", state).replace("_STATESHIFT_", to_str(stateshift)).replace("_COLOR_", colors[state_i % len(colors)])
		state_line = summary.format(state = state, stateshift = stateshift)
		template_summary += state_line

	template_summary=template_summary.replace('\n','\n\t')
	template_summary += "\n\treturn {"
	template_summary += ",".join(['"'+state+'"' ':' +state + "summary" for state in states])
	template_summary += '}, t, time_elapsed, solver_steps'

	target = open("templates/ode_python_dbmf.template", 'r')
	template = target.read()
	target.close()

	# execute python code in curly brackets
	template = template.replace(r'}',r'}{')
	template = template.split(r'{')
	for i in range(len(template)):
		if template[i].endswith(r'}'):
			template[i] = eval(template[i][:-1])
			template[i] = to_str(template[i])
	template = ''.join(template)

	target = open(model['output_path'], 'w')
	target.write(template)
	target.close()
	logger.info('Successfully created file.')


def generate_file_pa(lines, model):
	if not os.path.exists(model['output_dir']):
		os.makedirs(model['output_dir'])

	states = model['states']
	odes_per_state = model['odes_per_state']
	ode_text = ""
	for line in lines: #TODO kann weg?
		if line.endswith("=") or line.endswith("= "):
			line += "0"
		ode_text += "\t"+line+"\n"
	colors = ['b','g','r','c','m','y','k']
	template_summary = ""
	summary = "{state}summary = [0.0] * points\nfor i in range(points):\n\tfor j in range(odes_per_state):\n\t\tode_index = j + {stateshift}\n\t\t{state}summary[i] += sol[i,ode_index]*bin_distribution[j]\n"
	for state_i in range(len(states)):
		state = states[state_i]
		stateshift = state_i * odes_per_state
		state_line = summary.format(state = state, stateshift = stateshift)
		template_summary += state_line
		#template_summary += summary.replace("_STATE_", state).replace("_STATESHIFT_", to_str(stateshift)).replace("_COLOR_", colors[state_i % len(colors)])

	template_summary=template_summary.replace('\n','\n\t')
	template_summary += "\n\treturn {"
	template_summary += ",".join(['"'+state+'"' ':' +state + "summary" for state in states])
	template_summary += '}, t, time_elapsed, solver_steps'

	target = open("templates/ode_python_pa.template", 'r')
	template = target.read()
	target.close()

	# execute python code in curly brackets
	template = template.replace(r'}',r'}{')
	template = template.split(r'{')
	for i in range(len(template)):
		if template[i].endswith(r'}'):
			template[i] = eval(template[i][:-1])
			template[i] = to_str(template[i])
	template = ''.join(template)

	target = open(model['output_path'], 'w')
	target.write(template)
	target.close()
	logger.info('Successfully created file.')



#------------------------------------------------------
# Combinatorics
#------------------------------------------------------

def m_k_of(k, dim):
	l_old= {tuple([0]*dim)}
	l_new = set()
	len_old = -1
	while True:
		for l in l_old:
			for pos in range(dim):
				for i in range(k+1):
					l2 = list(l)
					l2[pos] = i
					l_new.add(tuple(l2))
		l_old = l_new.copy()
		#print(len(l_old))
		if len_old == len(l_old):
			break
		len_old = len(l_old)
	return {l for l in l_new if np.sum(l) == k}

def generate_neighbours(k_max, dim):
	neighbours = list()
	for k in range(k_max+1):
		for m in sorted(list(m_k_of(k, dim))):
			neighbours.append(m)
	return neighbours

def elemsin_k_vec_with_sum_m(k,m):
	#TODO check
	from scipy.special import binom
	return binom(m+k-1, k-1)

def multinomial_pmf(choice_vector, probability_vector):
	from scipy.stats import multinomial
	return multinomial.pmf(choice_vector, n=np.sum(choice_vector), p=probability_vector)


#------------------------------------------------------
# Symbolic symplification
#------------------------------------------------------

def ode_simplify(formula):
	original = formula
	logger.debug('get formula: {}'.format(formula))
	try:
		prefix = ""
		suffix = ""
		if "=" in formula:
			prefix = formula.split("=")[0]+"="
			formula = formula.split("=")[1]
			assert("=" not in formula)
		if "if" in formula:
			formula = formula.split("if")[0]
			suffix = "if"+formula.split("if")[1]
			assert("=" not in formula)

		formula = formula.replace("[","_B___O_").replace("]","_B___C_")
		formula = formula.replace("{","_CB___O_").replace("}","_CB___C_")
		new_formula = sympy.sympify(formula)
		new_formula = str(new_formula)
		new_formula = prefix + new_formula.replace("_B___O_", "[").replace("_B___C_","]").replace("_CB___O_","{").replace("_CB___C_","}")+suffix
		logger.debug('return converted formula: {}'.format(new_formula))
		return new_formula
	except:
		import sys
		logger.warn('could not convert formula: {} ({})'.format(formula,sys.exc_info()[0]))
		#print(formula, sys.exc_info()[0])
		return original


#------------------------------------------------------
# Compare models
#------------------------------------------------------

def compare_models(model1, model2): #TODO
	''' Computes difference of two models as the maximal L2 distance between points of their correspondig trajectories'''
	if 'trajectories' not in model1 or len(model1['trajectories']) == 0:
		raise ValueError('No trajectories in model to compare.')

	states = sorted(list(model1['trajectories'].keys()))
	if states != sorted(list(model2['trajectories'].keys())):
		raise ValueError('Cannot compare models, as they contain different states.')

	error_list = list()
	sample_num = len(model1['trajectories'][states[0]])

	for i in range(sample_num):
		error = 0.0
		for state in states:
			error += (model1['trajectories'][state][i] - model2['trajectories'][state][i])**2
		error_list.append(np.sqrt(error))

	return np.max(error_list)

#------------------------------------------------------
# Write files
#------------------------------------------------------

def write_trajectory_plot(models, filepath, show_plot = False, state_to_color = None):
	''' plots trajectories of one or two models, saves as .png and .svg, do not include filending in filepath arguement'''

	import matplotlib.pyplot as plt

	if not type(models) is list:
		models = [models]

	def state_to_color_default(state):
		state = state.lower().strip()
		color_dict = {'s': 'blue', 'i': plt.get_cmap('gnuplot')(0.45), 'r': 'green', 'ii': plt.get_cmap('gnuplot')(0.575), 'iii': plt.get_cmap('gnuplot')(0.7)}
		return color_dict.get(state, None)

	if state_to_color is None:
		state_to_color = state_to_color_default
	trajectories1 = models[0]['trajectories']
	subtitle = models[0]['name']+'(-)'
	plt.clf()
	for state in trajectories1:
		plt.plot(models[0]['time'], trajectories1[state], label=state, color = state_to_color(state), linewidth = 2)
	try:
		trajectories2 = models[1]['trajectories']
		subtitle += '    '+models[1]['name']+'(--)  '
		for state in trajectories2:
			plt.plot(models[1]['time'], trajectories2[state], label=state, ls='--', color = state_to_color(state), linewidth = 2)
	except IndexError:
		pass
	ncol = 2 #if len(models[0]['states']) > 3 else 1
	plt.legend(loc='best', ncol = ncol)
	plt.xlabel('t')
	plt.suptitle(subtitle)
	plt.grid()
	plt.savefig(filepath+'.png', dpi=300)
	plt.savefig(filepath+'.svg', format='svg', dpi=1200)
	if show_plot:
		plt.show()

def models_to_csv(models, filepath, header='sep=;\n', sep=';'):
	if not type(models) is list:
		models = [models]
	keys = set()
	for model in models:
		for key in model:
			if key not in ['trajectories', 'time']:
				keys.add(key)
				if sep in key:
					raise ValueError('Seperator sign in key.')
	keys = sorted(list(keys))
	with open(filepath,'w') as f:
		f.write(header)
		f.write(sep.join(keys)+'\n')
		line = list()
		for i in range(len(models)):
			line = list()
			for key in keys:
				rep = ',' if sep == ';' else ';'
				line.append(str(models[i].get(key,'')).replace(sep, rep).replace('\n',' --- '))
			f.write(sep.join(line))
			if i != list(range(len(models)))[-1]:
				f.write('\n')

def trajectories_to_csv(model, filepath, header='sep=;\n', sep=';'):
	trajectories = model['trajectories']
	time = model['time']
	# if folder != '' and not os.path.exists(folder):
	# 	os.makedirs(folder)
	# write csv
	with open(filepath, 'w') as f:
		states = sorted(list(trajectories.keys()))
		f.write(header)
		f.write('time'+sep+sep.join(states)+'\n')
		for i in range(len(trajectories[states[0]])):
			f.write(str(time[i])+sep)
			for state in states:
				s = sep if state != states[-1] else ''
				f.write(str(trajectories[state][i])+s)
			if i != list(range(len(trajectories[states[0]])))[-1]:
				f.write('\n')
