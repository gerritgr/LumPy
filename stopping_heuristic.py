import ame
import glob
from utilities import *
import sys
sys.dont_write_bytecode = True
import traceback
import pandas as pd
import matplotlib.pyplot as plt


xypoints = list()
def curve_(params):
	#TODO weight points with higher x less
	global xypoints
	c,k = params
	diff = 0.0
	for x,y in xypoints:
		y_pred = c*x**(-k)
		costs = (y_pred - y)**2
		#costs = costs * (1.0/y)
		diff += costs
	return diff

predicted_curves = list()
counter = 0
def predict_curve(x, y, max_cluster, model, eval_points=1000):
	global counter, xypoints, predicted_curves
	counter += 1
	countmax = np.max(x)+100000
	plt.clf()
	from scipy.optimize import minimize

	#activat to only use first two points
	#xypoints = list(zip(x,y))
	max_index = max(2, int(len(x)/2.0+0.5))
	xypoints = list(zip(x,y))[:max_index] #optional

	res = minimize(curve_,x0=[1.0,1.0], tol=1e-6)
	c,k = res.x
	logger.info('Fittet power law has parameters: c: {c}, k: {k}'.format(c=c, k=k))
	xplot = np.linspace(0,max_cluster, 1000)
	yplot = [c*x_i**(-k) if x_i >0.0 else 0.0 for x_i in xplot]
	#logger.info('yplot'+repr(yplot))

	df = pd.DataFrame({'clusternum': x, 'error' : y})
	df.to_csv(model['output_dir']+'fittet_curve_points_{}_{}.csv'.format(countmax, counter), header='sep=,')
	fig=plt.figure()
	ax=fig.add_subplot(111)
	ax.plot(xplot[1:], yplot[1:],alpha=0.3, ls='--')
	ax.scatter(x,y)
	ax.set_ylim([0.0, np.max(y)])
	plt.savefig(model['output_dir']+'fittet_curve_{}.pdf'.format(counter))
	predicted_curves.append(list(yplot))

	df = pd.DataFrame({'clusternum': xplot, 'error' : yplot})
	df.to_csv(model['output_dir']+'fittet_curve_full_{}_{}.csv'.format(countmax, counter), header='sep=,')
	return xplot, yplot

def predict_cluster(clustercounts, diffs, max_cluster, model, scaling = True):
	logger.info('Distances for different baselines (original):\t'+repr(list(zip(clustercounts, diffs))))
	ratio = max_cluster/clustercounts[-1]
	
	diffs = [d*np.power(ratio, 1/4.0) for d in diffs]
	#clustercounts = [d*np.power(ratio, 1/3.0) for d in clustercounts]

	#clustercounts[1] = clustercounts[1] * ratio**(1/10.0)
	#diffs = [d*np.power(ratio, 1/10.0) for d in diffs] #optional
	curve_x, curve_y = predict_curve(clustercounts, diffs, max_cluster, model)

	# x = numpy.linspace(0,10,1000)
	# dx = x[1]-x[0]
	# y = x**2 + 1
	# dydx = numpy.gradient(y, dx)


	curve_derivative = np.gradient(curve_y, curve_x[1]-curve_x[0]) #np.diff(curve_y)

	logger.info('Distances for different baselines (after scaling):\t'+repr(list(zip(clustercounts, diffs))))
	first_index = -1

	df = pd.DataFrame({'curve_x': curve_x, 'curve_derivative' : curve_derivative})
	df.to_csv(model['output_dir']+'derivative_{}.csv'.format(clustercounts[-1]), header='sep=,')
	plt.clf()
	fig=plt.figure()
	ax=fig.add_subplot(111)
	ax.plot(curve_x[10:], curve_derivative[10:])
	ax.set_ylim([np.min(curve_derivative),np.max(curve_derivative)])
	plt.savefig(model['output_dir']+'derivative_{}.pdf'.format(clustercounts[-1]))

	if 'derivative_threshold' in model:
		# THRESHOLD = 0.000015
		THRESHOLD = float(model['derivative_threshold'])
		for i, value in enumerate(curve_derivative):
			if np.abs(value) < np.abs(THRESHOLD) and i>0:
				return curve_x[i]
	elif 'stopping_interval' in model:
		#THRESHOLD = 0.95
		THRESHOLD = float(model['stopping_interval'])
		y_values = np.interp(curve_x, clustercounts, diffs)
		z = np.sum(y_values)
		y_values = [y/z for y in y_values]
		s = 0.0
		final_i = None
		for i, y in enumerate(y_values):
			s+=y
			if s>THRESHOLD:
				final_i=i
				break
		logger.info('index where threshold is rached is: '+str(final_i))
		return final_i * 1/ratio
	else:
		THRESHOLD = 0.01 if model is None or 'stopping_threshold' not in model else float(model['stopping_threshold'])
		for i, value in enumerate(curve_y):
			if np.abs(value) < THRESHOLD and i>0:
				return curve_x[i]
	logger.warn('No point found.')
	return curve_x[-1]

def find_new_clusternum(solved_models, max_cluster):
	basemodel = solved_models[-1]
	diffs = list()
	clustercounts = list()
	for m in solved_models[:-1]:
		diffs.append(compare_models(m, basemodel))
		clustercounts.append(m['actual_cluster_number'])
	diffs.append(0.0)
	clustercounts.append(basemodel['actual_cluster_number'])
	predict_cluster_number = predict_cluster(clustercounts, diffs, max_cluster, basemodel)
	return predict_cluster_number

def curve_distance(curve1, curve2):
	assert(len(curve1) == len(curve2))
	dist = 0.0
	for i, _ in enumerate(curve1):
		dist += np.abs(curve1[i] - curve2[i])
	dist = dist/len(curve1)
	logger.info('Distance between last two predicted error curves is: {}'.format(dist))
	return dist

def stopping_heuristic(modelpath):
	global predicted_curves
	import ClusterEngine as CE
	solved_models = list()
	actual_clusters = list()
	actual_degrees = list()
	base = read_model(modelpath)
	k_max = base['k_max']
	if k_max < 15:
		logger.error('k_max should be larger than 15.')
		return
	#first_tries =  [6, 10, 12]
	first_tries =  [6, 10, 12]
	max_cluster = None


	for base_num in first_tries:
		m1 = read_model(modelpath)
		m1['bin_num'] = base_num
		m1['name'] += 'B'+str(base_num)
		logger.info('Try base model {}'.format(m1['name']))
		ame.generate_and_solve(m1, True, False)
		solved_models.append(m1.copy())
		actual_clusters.append(m1['actual_cluster_number'])
		actual_degrees.append(base_num)
		max_cluster = m1['max_cluster']
		old_k = base_num

	while True:
		new_k = None
		logger.info('Find cluster number for next shot')
		min_cluster_num = find_new_clusternum(solved_models, max_cluster)
		logger.info('Lower bound for next clutering based soley on curve is: {}'.format(min_cluster_num))
		if min_cluster_num <= actual_clusters[-1]:
			logger.info('stop with lower bound for cluster number: {}'.format(min_cluster_num))
			break
		logger.info('Lower bound for next clutering is {}'.format(min_cluster_num))
		#logger.info('Predicted error at this point is: {}, predicted gradient is: {}'.format())
		if min_cluster_num >= max_cluster:
			# hack to artifically get unclustered version
			old_k=max_cluster
		k = old_k
		while True: # for k in np.arange(old_k + 1, k_max): # note that k can be larger than k_max
			k = k+1
			m1 = read_model(modelpath)
			m1['bin_num'] = k
			#m1['name'] += 'B'+str(k)
			logger.info('Check cluster number generated by {} inter-degree clusters'.format(k))
			clustering = CE.clustering(m1)
			new_k = k
			len_clustering = len(set(clustering.values()))
			logger.info('{} inter-degree clusters generate {} clusters in total.'.format(k, len_clustering))
			if len_clustering >= min(min_cluster_num, max_cluster):
				logger.info('{} inter-degree clusters are enough'.format(k))
				k = k+1 # to avoid repetitive unter-aproximation
				new_k = k
				logger.info('Actually we will use k= {} '.format(k))
				m1['bin_num'] = k
				m1['name'] += 'B'+str(k)
				break
		ame.generate_and_solve(m1, True, False)
		old_k = new_k
		solved_models.append(m1.copy())
		actual_clusters.append(m1['actual_cluster_number'])
		actual_degrees.append(new_k)
		if actual_clusters[-2] == actual_clusters[-1]:
			# this should not happen
			break
	logger.info('Stop heuristic with inter-degree bins: {}, and cluster per equation: {}'.format(actual_degrees[-1], actual_clusters[-1]))



#------------------------------------------------------
# Stopping for DBMF/PA
#------------------------------------------------------

import pa
import dbmf
from utilities import *

AUTO_BIN_INCREASE_START = 10	#number of bins to start with
AUTO_BIN_INCREASE_STOP = 0.0010 	#diff to previouse call, stopping criterion

def find_next(diff1, diff2, current_bin_num, on_error_increase = 10):
	if on_error_increase <= 0:
		raise ValueError('on_error_increase must be positive.')
	if current_bin_num <= 0:
		raise ValueError('current_bin_num cannot be 0.')
	step_size_scale = 2000
	slope = diff1 - diff2
	if slope <= 0.0:
		logger.info('differences are getting larger')
		return current_bin_num + on_error_increase
	next_step = int(current_bin_num + slope * step_size_scale + 0.5)
	logger.info('Next step (untruncated) is {}'.format(next_step))
	return next_step

def solve_model_autobin(model, method='PA'):
	assert(method in ['PA', 'DBMF'])
	if method == 'PA':
		solve_model = pa.solve_model
		run_model = pa.run_model
	else:
		solve_model = dbmf.solve_model
		run_model = dbmf.run_model

	input_model = model
	logger.info('Tries to determine number of bins automatically.')
	max_bin = len([p for p in model['degree_distribution'] if not isclose(p, 0.0)])

	if AUTO_BIN_INCREASE_START + 10 >= max_bin:
		raise ValueError('AUTO_BIN_INCREASE_START cannot be larger/equal than/to possible number of bins.')

	possible_bin_numbers = [AUTO_BIN_INCREASE_START, AUTO_BIN_INCREASE_START + 5, AUTO_BIN_INCREASE_START + 10]
	models = None
	diff_1_to_2 = None
	diff_2_to_3 = None
	while True:
		models = list()

		for bin_num in possible_bin_numbers:
			logger.info('Tries number of bins: '+str(bin_num))
			current_model = model.copy()
			current_model['bin_num'] = bin_num
			current_model['name'] = current_model['name'].split('_B_')[0]+'_B_'+str(current_model['bin_num'])
			model = solve_model(current_model, call_after_creation = False)
			model = run_model(model, suppress_output = True)
			models.append(model.copy())
			if len(models) == 2:
				diff_1_to_2 = compare_models(models[0], models[1])
				# for different stopping criterion
				# if diff_1_to_2 < AUTO_BIN_INCREASE_STOP:
				# 	logger.info('found bin number '+str(model['bin_num']))
				# 	logger.info('final result is '+str(model['output_path']))
				# 	break

		if len(models) == 1:
			logger.info('stop at maximal bin number.')
			logger.info('final result is '+str(model['output_path']))
			break
		if len(models) == 2:
			break

		diff_1_to_2 = compare_models(models[0], models[1])
		diff_2_to_3 = compare_models(models[1], models[2])
		if np.isnan(diff_1_to_2) or np.isnan(diff_2_to_3):
			logger.error(model_to_str(models[0]))
			logger.error(model_to_str(models[1]))
			logger.error('Difference is nan.')
		logger.info('difference between {m0} and {m1} is {d12}'.format(m0 = models[0]['name'], m1 = models[1]['name'], d12 = diff_1_to_2))
		logger.info('difference between {m1} and {m2} is {d23}'.format(m1 = models[1]['name'], m2 = models[2]['name'], d23 = diff_2_to_3))

		# for different stopping criterion
		# if diff_2_to_3 < AUTO_BIN_INCREASE_STOP:
		# 	logger.info('found bin number '+str(model['bin_num']))
		# 	logger.info('final result is '+str(model['output_path']))
		# 	break

		slope = diff_1_to_2 - diff_2_to_3
		if slope > 0.0 and slope < AUTO_BIN_INCREASE_STOP:
			logger.info('found bin number '+str(model['bin_num']))
			logger.info('final result is '+str(model['output_path']))
			break

		new_point = find_next(diff_1_to_2, diff_2_to_3, possible_bin_numbers[-1])
		new_point = np.max([new_point, possible_bin_numbers[-1]+5]) #minimal step size is 5
		new_point = np.min([new_point, possible_bin_numbers[-1]+30]) #maximal step size is 30
		if new_point+10 >= max_bin:
			possible_bin_numbers = [max_bin]
			continue
		logger.info('next bin number to evaluate is {}'.format(new_point))

		possible_bin_numbers = [new_point, new_point + 5, new_point + 10]

	model = models[-1]
	logger.info('stopped searching for bin number, save results...')
	filepath = model['output_path'][:-3] if model['output_path'].endswith('.py') else  model['output_path']
	write_trajectory_plot(model, filepath+'_auto')
	trajectories_to_csv(model, filepath+'_auto.csv')

	for m in model:
		input_model[m] = model[m]
	logger.info('done')


def solve_model_autobin_error(model, bin_num_start=3, update=lambda x: x+1, acceptable_error=0.003, method = 'PA'):
	''' Tries to find the smallest number of bins resulting in an error <=  acceptable_error.
	The error is from the comparision to the input model (and its corrsponding number of bins).
	Naturally, to calculate the error the whole model has to be solved.'''

	assert(method in ['PA', 'DBMF'])
	if method == 'PA':
		solve_model = pa.solve_model
		run_model = pa.run_model
	else:
		solve_model = dbmf.solve_model
		run_model = dbmf.run_model

	logger.info('Solve model and find number of bin for given error.')
	base = solve_model(model, call_after_creation = False)
	base = run_model(base, suppress_output = True)
	logger.info('Solved baseline model successfully.')

	possible_bin_numbers = [bin_num_start]
	while possible_bin_numbers[-1] != base['bin_num']:
		possible_bin_numbers.append(min(update(possible_bin_numbers[-1]), base['bin_num']))
	#logger.debug('Possible bin numbers are: '+str(possible_bin_numbers))

	if bin_num_start >= base['bin_num'] or len(possible_bin_numbers) <= 1:
		raise ValueError('No number of bins to test against, try a smaller bin_num_start or a larger number of bins for the baseline.')

	for bin_num in possible_bin_numbers:
		logger.info('Solve model with '+str(bin_num)+' bins.')
		test = base.copy()
		test['bin_num'] = bin_num
		test['name'] = test['name'].split('_B_')[0]+'_B_'+str(test['bin_num'])
		test = solve_model(test, call_after_creation = False)
		test = run_model(test, suppress_output = True)
		logger.info('Solved test model successfully.')
		diff = compare_models(base, test)
		logger.info('Diffenrece to baseline is '+str(diff))
		if diff < acceptable_error:
			logger.info('Set number of bins to:' + str(bin_num))
			break

	filepath = base['output_path'][:-3] if base['output_path'].endswith('.py') else base['output_path']
	filepath += '_'+str(bin_num)
	write_trajectory_plot([test, base], filepath+'_autoe')
	trajectories_to_csv(base, filepath+'autoebase.csv')

	filepath = test['output_path'][:-3] if test['output_path'].endswith('.py') else test['output_path']
	#write_trajectory_plot(test, filepath)
	trajectories_to_csv(test, filepath+'_autoetest.csv')

def compare_errortypes(model, bin_num_start=10, update=lambda x: x+5, bin_num_stop = 81, write_time = True, method='PA'):
	''' Only for evaluation purposes. Compares the actual error (difference to the umbinned model) to the difference
	to the previsouse model. That there is some kind of relation between the two types of errors is the reason
	solve_model_autobin_error() makes sense.
	'''
	assert(method in ['PA', 'DBMF'])
	if method == 'PA':
		solve_model = pa.solve_model
		run_model = pa.run_model
	else:
		solve_model = dbmf.solve_model
		run_model = dbmf.run_model

	import matplotlib
	matplotlib.use('agg')
	import matplotlib.pyplot as plt
	assert(bin_num_stop <= model['bin_num'])

	logger.info('Generate baseline model.')
	base = solve_model(model, call_after_creation = False)
	logger.info('Solve baseline model.')
	base = run_model(base, suppress_output = True)
	previouse_model = None
	diff_to_baseline_list = list()
	diff_to_previouse_list = list()
	bin_num_list = list()
	current_bin_num = bin_num_start

	while True:
		if current_bin_num > bin_num_stop:
			logger.info('Done testing.')
			break
		logger.info('Test number of bins: '+str(current_bin_num))
		test = base.copy()
		test['bin_num'] = current_bin_num
		test['name'] = test['name'].split('_B_')[0]+'_B_'+str(test['bin_num'])
		test = solve_model(test, call_after_creation = False)
		test = run_model(test, suppress_output = True)
		diff_to_baseline = compare_models(base, test)
		diff_to_previouse = -1
		if not previouse_model is None:
			diff_to_previouse = compare_models(test, previouse_model)
		logger.info('Diff to baseline is: {}'.format(diff_to_baseline))
		logger.info('Diff to previouse model is: {}'.format(diff_to_previouse))
		diff_to_baseline_list.append(diff_to_baseline)
		diff_to_previouse_list.append(diff_to_previouse)
		bin_num_list.append(current_bin_num)
		current_bin_num = update(current_bin_num)
		previouse_model = test
	logger.info('Diff to base list: '+str(diff_to_baseline_list))
	logger.info('Diff to prev list: '+str(diff_to_previouse_list))

	filepath = model['output_path'][:-3] if model['output_path'].endswith('.py') else model['output_path']


	import numpy as np
	import matplotlib as mpl
	import matplotlib.pyplot as plt
	plt.clf()
	mpl.rc('xtick', labelsize=13)
	mpl.rc('ytick', labelsize=13)
	alpha = 0.6
	y_label = 'Error'
	x_label = 'Number Of Bins'
	#plt.suptitle('Blue is diff to baseline, red is diff to previouse run. '+model['name'])
	area = 150.0
	fig, ax1 = plt.subplots()
	ax1.set_xlabel(x_label,fontsize=13)
	ax1.set_xlim([0, 82])
	fig, ax1 = plt.subplots()
	ax1.set_xlabel(x_label,fontsize=13)
	ax1.set_xlim([0, 82])
	s1 = ax1.scatter(bin_num_list[1:], diff_to_baseline_list[1:], s=area, c='b', alpha=alpha, marker = '*')
	s2 = ax1.scatter(bin_num_list[1:], diff_to_previouse_list[1:], s=area, c='r', alpha=alpha, marker = '.')
	ax1.set_ylabel(y_label,fontsize=13)
	ax1.set_ylim([0, np.max(diff_to_baseline_list+diff_to_previouse_list)*1.01])
	plt.savefig(filepath+'_errorscompare.pdf', format='pdf', bbox_inches='tight')

	plt.clf()
	fig, ax1 = plt.subplots()
	ax1.set_yscale('log')
	ax1.set_xscale('log')
	s1 = ax1.scatter(bin_num_list[1:], diff_to_baseline_list[1:], s=area, c='b', alpha=alpha, marker = '*')
	s2 = ax1.scatter(bin_num_list[1:], diff_to_previouse_list[1:], s=area, c='r', alpha=alpha, marker = '.')
	ax1.set_ylim([0, 1.3 * np.max(diff_to_baseline_list[1:]+diff_to_previouse_list[1:])])
	ax1.set_xlim([0, 1.3 * bin_num_list[-1]])
	ax1.set_xlabel(x_label,fontsize=13)
	ax1.set_ylabel(y_label,fontsize=13)
	plt.savefig(filepath+'_errosloglogcompare.pdf', format='pdf', bbox_inches='tight')

	# plt.clf()
	# plt.suptitle('Blue is diff to baseline, red is diff to previouse run. '+model['name'])
	# plt.scatter(bin_num_list[1:], diff_to_baseline_list[1:], s = 60, marker = '^', c='blue', alpha=0.9)
	# plt.scatter(bin_num_list[1:], diff_to_previouse_list[1:], s = 60, marker = '*', c='red', alpha=0.9)
	# plt.ylim(0, 1.1 * np.max(diff_to_baseline_list[1:]+diff_to_previouse_list[1:]))
	# plt.xlim(0, 1.1 * bin_num_list[-1])
	# plt.savefig(filepath+'_errors.pdf')

	model['bin_num_list'] = bin_num_list[1:]
	model['diff_to_previouse_list'] = diff_to_previouse_list[1:]
	model['diff_to_baseline_list'] = diff_to_baseline_list[1:]
	models_to_csv(model, filepath+'_errors.csv')
