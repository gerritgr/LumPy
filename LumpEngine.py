import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import sys, os, glob, subprocess
from utilities import *
#from sympy import *
from sympy import simplify


def multi_replace(s, subs):
	for x,y in subs:
		s = s.replace(x,y)
	return s

def clean(s):
	s = s.replace('["', '___open_bracket___')
	s = s.replace('"]', '___open_closed___')
	s = s.replace('[', '___open_bracket2___')
	s = s.replace(']', '___open_closed2___')
	return s

def declean(s):
	s = s.replace('___open_bracket___', '["')
	s = s.replace('___open_closed___', '"]')
	s = s.replace('___open_bracket2___', '[')
	s = s.replace('___open_closed2___', ']')
	return s

def to_symbols(s):
	try:
		from symengine import sympify
		return sympify(s)
	except:
		from sympy import sympify
		return sympify(s)

def to_symengine(s):
	from symengine import var, sympify
	s = str(s)
	seperators = " ,;,+,-,=,),(,*,/,**".split(',')
	for ch in seperators:
		s = s.replace(ch, ' '+ch+' ')
	s_list = s.split(' ')
	v_list = [token for token in s_list if '_' in token]
	v_list = list(set(v_list))
	v = ' '.join(v_list)
	if v.strip() != '':
		var(v)
	return sympify(eval(s))


def compute_formula_partial(line, subs=None):
	global substitution
	if len(line['ode_formula']) == 0:
		return '0'
	s = to_symbols('0')
	formula = line['ode_formula']
	weight = line['weight_normalized']
	if subs is None:
		subs = substitution
	for i in range(len(formula)):
		s += to_symbols('({w}*({f}))'.format(w=weight[i], f=multi_replace(clean(formula[i]),subs)))
	s = str(s)
	s = declean(s)
	return s

def compute_formula(line, subs=None):
	if subs is None:
		subs = substitution
	if len(line['ode_formula']) == 0:
		return '0'
	s = '0'
	formula = line['ode_formula']
	weight = line['weight_normalized']
	for i in range(len(formula)):
		s += '+({w}*({f}))'.format(w=weight[i], f=multi_replace(clean(formula[i]),subs))
	s = to_symbols(s)
	s = str(s)
	s = declean(s)
	return s

def compute_init(line):
	weight = line['weight_normalized']
	init = line['initial_value']
	init_mean = 0.0
	for i in range(len(init)):
		init_mean += weight[i]*init[i]
	return init_mean

def compute_degree_count(line):
	weight = line['weight_normalized']
	degree_count = line['degree_count']
	degree_count_mean = 0.0
	for i in range(len(weight)):
		degree_count_mean += weight[i]*degree_count[i]
	return degree_count_mean

def compute_subs(df, scale_during_substitution):
	subs = list()
	#scale_during_substitution=False
	if scale_during_substitution:
		for _, row in df.iterrows():
			old_names = row['old_names']
			old_inits = row['initial_value_old']
			old_degrees = row['degree_count']
			cluster_name = row['ode_name']
			init = row['initial_value']
			avg_degree = row['degree_count_avg']
			for i in range(len(old_names)):
				old_name = old_names[i]
				old_init = old_inits[i]
				init_scale = old_init/init
				init_scale = avg_degree/old_degrees[i]
				new_name_dt =  '({s}*{n})'.format(s=init_scale, n = clean(cluster_name))
				new_name = '({s}*{n})'.format(s=init_scale, n = clean(cluster_name)[3:])
				subs.append((clean(old_name), new_name_dt))
				subs.append((clean(old_name)[3:], new_name))

	else:
		for _, row in df.iterrows():
			old_names = row['old_names']
			cluster_name = row['ode_name']
			for name in old_names:
				subs.append((clean(name), clean(cluster_name)))
				subs.append((clean(name)[3:], clean(cluster_name)[3:]))
	return subs

def compute_formula_torow(df):
	result = df.apply(compute_formula, axis=1)
	logger.debug('Pool done.')
	return result

substitution = dict()
def lump(df, scale_during_substitution = True):
	global substitution
	from functools import partial
	assert('ode' in df.columns and 'cluster_indicator' in df.columns and 'initial_value' in df.columns)
	assert('name_old' not in df.columns)
	assert('old_names' not in df.columns)
	assert('ode_formula' not in df.columns)
	assert('ode_name' not in df.columns)
	assert('weight_normalized' not in df.columns)
	assert('weight_sum' not in df.columns)
	assert('initial_value_old' not in df.columns)
	#assert(normalization in ['standard', 'softmax'])
	if 'weight' not in df.columns:
		df['weight'] = 1.0
	#df = df[df['weight'] > 0]
	df['ode_name'] = df.apply(lambda l: l['ode'].split('=')[0].strip(), axis=1)
	df['ode_formula'] = df.apply(lambda l: l['ode'].split('=')[1].strip(), axis=1)
	del df['ode']
	df = df.groupby('cluster_indicator')
	df = df.agg(lambda l: tuple(l))
	df = df.reset_index()
	df['old_names'] = df['ode_name']
	df['ode_name'] = ['dt_x["cluster'+str(i).zfill(len(str(len(df['old_names']))))+'"]' for i in range(len(df['old_names']))]
	df['weight_sum'] = df.apply(lambda l: np.sum(l['weight']), axis=1)
	df['weight_normalized'] = df.apply(lambda l: tuple([v/l['weight_sum'] for v in l['weight']]), axis=1)
	df['initial_value_old'] = df['initial_value']
	df['initial_value'] = df.apply(compute_init, axis=1)
	df['degree_count_avg'] = df.apply(compute_degree_count, axis=1)

	# parallel lumping
	substitution = compute_subs(df, scale_during_substitution)
	logger.info('Compute lumped ODEs.')
	from multiprocessing import Pool, cpu_count
	cores = cpu_count()
	data_split = np.array_split(df, len(df.index))
	pool = Pool(cores)
	new_data = pool.map(compute_formula_torow, data_split)
	data = pd.concat(new_data)
	pool.close()
	pool.join()
	df['ode_formula'] = data
	del new_data
	del data_split
	del data
	# same as
	# same as df['ode_formula'] = df.apply(lambda x: compute_formula(x,substitution), axis=1)
	logger.info('Compute lumped ODEs Done.')

	df['ode'] = df['ode_name']+'='+df['ode_formula']
	return df

# d = {'cluster_indicator': [2,3,2], 'formula':['3+x[0]','4+y','5+x**2+z'], 'name':['x','y','z'], 'weight': [0.6,1.4,0.4]}
# df = pd.DataFrame.from_dict(d)
# dx = lump(df)
# print(dx)
