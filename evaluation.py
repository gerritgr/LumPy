import ame
import glob
from utilities import *
import sys
import time
sys.dont_write_bytecode = True
import traceback
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import simulation as mc
matplotlib.use('agg')	#run without an X-server
import pa, dbmf
from stopping_heuristic import *
import time


def plot_scatter(x,y, outpath, color = 'r', x_label = 'Cluster Number per State', y_label = 'Distance to Original Equation', set_lim = True):
	import numpy as np
	import matplotlib
	matplotlib.use('agg')	#run without an X-server
	import matplotlib as mpl
	import matplotlib.pyplot as plt
	mpl.rc('xtick', labelsize=13)
	mpl.rc('ytick', labelsize=13)
	alpha = 0.6
	area = 170.0
	fig, ax1 = plt.subplots()
	ax1.set_xlabel(x_label,fontsize=14)
	if set_lim: ax1.set_xlim([0, max(x)*1.1])
	s1 = ax1.scatter(x, y, s=area, c=color, alpha=alpha, marker = '*')
	ax1.set_ylabel(y_label,fontsize=14)
	if set_lim: ax1.set_ylim([0, max(y)*1.1])
	plt.savefig(outpath, format='pdf', bbox_inches='tight')
	df = pd.DataFrame({x_label: x, y_label : y})
	df.to_csv(outpath[:-4]+'.csv', header='sep=,')
	plt.close()

def plot_sol_distances(model_list):
	#print('modellist', [m.keys() for m in model_list])
	base = model_list[0]
	distances = dict()
	for model1_i in range(len(model_list)):
		for model2_i in range(len(model_list)):
			if model1_i <= model2_i:
				m1 = model_list[model1_i]
				m2 = model_list[model2_i]
				clusternum1 = m1['actual_cluster_number']
				clusternum2 = m2['actual_cluster_number']
				if model1_i == model2_i:
					distances[(clusternum1,clusternum2)] = 0.0
					continue
				if clusternum1 > clusternum2:
					distances[(clusternum1,clusternum2)] = compare_models(m1, m2)
				else:
					distances[(clusternum2,clusternum1)] = compare_models(m1, m2)

	coordinates = distances.keys()
	print('coordinates', coordinates)
	x = [a for a,b in coordinates]
	y = [b for a,b in coordinates]
	color = [distances[cord] for cord in coordinates]
	import numpy as np
	import matplotlib
	matplotlib.use('agg')	#run without an X-server
	import matplotlib as mpl
	import matplotlib.pyplot as plt
	import seaborn as sns
	df = pd.DataFrame({'Number of Clusters per Equation X': x, 'Number of Clusters per Equation Y' : y, 'Value' : color})
	df.to_csv(base['output_path'][:-4]+'_distancematrix.csv', header='sep=,')
	plt.clf()
	sns.axes_style("white")
	fig, ax1 = plt.subplots()
	heatmap = np.zeros([int(max(x+y)+1.1), int(max(x+y)+1.1)])
	mask = np.zeros([int(max(x+y)+1.1), int(max(x+y)+1.1)])
	for i in range(int(np.max(x)+1.1)):
		for j in range(int(np.max(x)+1.1)):
			heatmap[i,j] = np.nan
			mask[i,j] = True
	for a,b in coordinates:
		heatmap[a,b] = distances[(a,b)]
		mask[a,b] = False
	#mask = heatmap.isnull()
	np.savetxt(base['output_path'][:-4]+'_distancematrix2.out', heatmap, delimiter=',')
	ax = sns.heatmap(np.random.random([int(np.max(x)+1.1),int(np.max(x)+1.1)]), cmap="YlGnBu", mask=mask, ax=ax1, square=True, xticklabels=20, yticklabels=20)
	#plt.show()
	#plt.savefig(base['output_path'][:-4]+'_distancematrix2.pdf', format='pdf', bbox_inches='tight')



def evaluate_model(modelpath, method, base = None):
	errors = list()
	bins = list()
	cluster_score = list()
	times = list()
	model_list = list()

	sns.set_style("white")

	mpl.rc('xtick', labelsize=19)
	mpl.rc('ytick', labelsize=19)
	alpha = 0.6
	area = 220.0
	fig, ax1 = plt.subplots()
	plt.axis('equal')
	ax1.spines["left"].set_visible(False)
	ax1.spines["top"].set_visible(False)
	ax1.spines["right"].set_visible(False)
	ax1.spines["bottom"].set_visible(False)
	ax1.set_xlabel('Number of Clusters per Equation',fontsize=14)
	ax1.set_xlim([0, max(x+y)*1.1])
	ax1.set_ylim([0, max(x+y)*1.1])
	s1 = ax1.scatter(x, y, s=area, c=color, alpha=alpha, marker = ',', cmap='viridis', edgecolors='black') # YlGnBu
	ax1.set_ylabel('Number of Clusters per Equation',fontsize=14)
	ax1.set_ylim([0, max(x+y)*1.1])
	ax1.set_xlim([0, max(x+y)*1.1])
	plt.colorbar(s1)
	plt.xticks(sorted(list(set(y))))
	plt.yticks(sorted(list(set(y))))
	plt.axis('equal')
	plt.savefig(base['output_path'][:-4]+'_distancematrix.pdf', format='pdf', bbox_inches='tight')
	df = pd.DataFrame({'Number of Clusters per Equation X': x, 'Number of Clusters per Equation Y' : y, 'Value' : color})
	df.to_csv(base['output_path'][:-4]+'_distancematrix.csv', header='sep=,')
	plt.clf()
	sns.axes_style("white")
	fig, ax1 = plt.subplots()
	heatmap = np.zeros([int(max(x+y)+1.1), int(max(x+y)+1.1)])
	mask = np.zeros([int(max(x+y)+1.1), int(max(x+y)+1.1)])
	for i in range(int(np.max(x)+1.1)):
		for j in range(int(np.max(x)+1.1)):
			heatmap[i,j] = np.nan
			mask[i,j] = True
	for a,b in coordinates:
		heatmap[a,b] = distances[(a,b)]
		mask[a,b] = False
	#mask = heatmap.isnull()
	np.savetxt(base['output_path'][:-4]+'_distancematrix2.out', heatmap, delimiter=',')
	ax = sns.heatmap(np.random.random([int(np.max(x)+1.1),int(np.max(x)+1.1)]), cmap="YlGnBu", mask=mask, ax=ax1, square=True, xticklabels=20, yticklabels=20)
	#plt.show()
	#plt.savefig(base['output_path'][:-4]+'_distancematrix2.pdf', format='pdf', bbox_inches='tight')



def evaluate_model(modelpath, method, base = None):
	global runtimes
	errors = list()
	bins = list()
	cluster_score = list()
	times = list()
	model_list = list()

	if base is None:
		base = read_model(modelpath)
		logger.info('evaluate with baseline')
		base['name'] += 'baseline'
		start = time.clock()
		ame.generate_and_solve(base, True, True)
		runtimes[(modelpath, 'baseline')] = time.clock() - start
	model_list.append(base.copy())

	for base_num in [i + 5 for i in range(10)] + [(i*3)+15 for i in range(40)]: #sparse evaluation at the end
	#for base_num in [i*3 + 5 for i in range(20)]:
		logger.info('evaluate with {} bins'.format(base_num))
		m1 = read_model(modelpath)
		m1['bin_num'] = base_num
		m1['merge'] = base_num
		m1['heuristic'] = m1['heuristic'] if method is None else method
		m1['name'] += 'B'+str(base_num)+'V'+m1['heuristic']
		start = time.clock()
		ame.generate_and_solve(m1, True, False)
		runtimes[(modelpath, 'bins_'+str(m1['actual_cluster_number']))] = time.clock() - start
		write_runtime()
		errors.append(compare_models(m1, base))
		bins.append(m1['actual_cluster_number'])
		#cluster_score.append(m1['cluster_score'])
		times.append(m1['time_elapsed'])
		# when you need to plot the sol distances
		#model_list.append(m1.copy())
		try:
			plot_sol_distances(model_list)
		except:
			pass
		#base_surrogate = model_list[-1]['loss_list'][0]
		plot_scatter(bins + [base['actual_cluster_number']],errors + [0.0], base['output_path'][:-4]+'_'+method+'_errorplot.pdf')
		plot_scatter(bins + [base['actual_cluster_number']],times + [base['time_elapsed']], base['output_path'][:-4]+'_'+method+'_timeplot.pdf', color='b', y_label = 'Time (s)')
		#plot_scatter(bins + [base['actual_cluster_number']],cluster_score + [base['cluster_score']], base['output_path'][:-4]+'_'+method+'_clusterscore.pdf', set_lim = False, color='y')

		#stop
		if base['actual_cluster_number'] == bins[-1]:
			logger.info('maximal cluster number is reached.')
			break

	#lot_sol_distances(model_list)





def evaluate_methods(modelpath, methods):
	global runtimes
	base = read_model(modelpath)
	base['name'] += 'baseline'
	start = time.clock()
	ame.generate_and_solve(base, True, True)
	runtimes[(modelpath, 'baseline_multi')] = time.clock() - start
	for method in methods:
		try:
			#for single threading
			#evaluate_model(modelpath, method, base)
			start_process(evaluate_model, (modelpath, method, base))
		except Exception as e:
			logger.error('Error during evaluation of model {mo} with method {me}: {e}'.format(mo=modelpath, me=methods, e=e))
			logger.error(traceback.format_exc())
	join_processes()


#stopping_heuristic('model/SIS50.model')
#evaluate_model('model/SIS50.model','cluster_subspaceXY')
#x=0/0

runtimes = dict()
def write_runtime():
	global runtimes
	with open('runtimeLog.txt', 'a') as f:
		f.write('----------------\n')
		for key,value in runtimes.items():
			logger.info(repr(key)+'\t'+repr(value)+'\n')
			f.write(repr(key)+'\t'+repr(value)+'\n')

def analyze_model(modelpath):
	global runtimes
	try:
		plt.clf()
		logger.info('analyze: \t'+modelpath)
		start = time.clock()
		dbmf.main(modelpath, True, True)
		runtimes[(modelpath, 'dbmf')] = time.clock() - start
		start = time.clock()
		pa.main(modelpath, True, True)
		runtimes[(modelpath, 'pa')] = time.clock() - start
		start = time.clock()
		mc.main(read_model(modelpath), 5, 1000, 95, None)
		runtimes[(modelpath, 'mcsimple')] = time.clock() - start

		start = time.clock()
		stopping_heuristic(modelpath)
		runtimes[(modelpath, 'stoppingheuristic')] = time.clock() - start
		write_runtime()
		#evaluate_methods(modelpath, ['cluster_subspaceXX', 'cluster_subspaceXY','cluster_subspaceXZ'])
		#return
		evaluate_model(modelpath,'cluster_subspaceXY')
		
		#evaluate_model(modelpath,'cluster_subspaceXY')
		write_runtime()
		#evaluate_model(modelpath,'cluster_subspaceXZ')
		simu_model = read_model(modelpath)
		start = time.clock()
		mc.main(simu_model, 10, 10000, 95, None)
		runtimes[(modelpath, 'mcfull')] = time.clock() - start

		logger.info('RUNTIME:\t'+repr(runtimes))
		write_runtime()
	except:
		write_runtime()
		logger.error('ERROR AT MODEL: \t'+modelpath)
		logger.error(traceback.format_exc())


def analyze_model_multi(modelpath, parallel = False):
	if parallel:
		start_process(analyze_model, (modelpath, ))
	else:
		analyze_model(modelpath)

import sys
pattern = 'model/*model'
try: 
    pattern = str(sys.argv[1])
except:
    pass
		
for modelpath in sorted(glob.glob(pattern)):
	analyze_model_multi(modelpath, parallel=False)
join_processes()
