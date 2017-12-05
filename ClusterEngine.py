import model_parser
import numpy as np
from utilities import *
import sys
import sympy
import time
import pandas as pd
import sys


def indicators_to_dict(indicators, odes):
	results = dict()
	for i in range(len(odes)):
		m = odes[i]
		results[m] = indicators[i]
	return results

def clustering(model):
	degree_distribution = model['degree_distribution']
	cluster_number =  model['bin_num']
	if 'neighborhood' not in model:
		model['neighborhood'] = list(generate_neighbours(model['k_max'], len(model['states'])))
	m_vectors = model['neighborhood']
	cluster_method = model['heuristic']
	logger.info('Start Clustering.')
	if cluster_number >= len(m_vectors):
		result = dict()
		for i, m in enumerate(m_vectors):
			result[m] = i
		return result
	#if cluster_number == 1:
	#	return {m:0 for m in m_vectors}
	#assert(cluster_number > 1)
	if isinstance(cluster_method, str):
		cluster_method = globals()[cluster_method]
	#try:
	#	result =  cluster_method(degree_distribution, cluster_number, m_vectors)
	#except:
	result =  cluster_method(model)
	logger.info('Clustering Done.')
	return result


def cluster(degree_distribution, cluster_number, m_vectors):
	from scipy.spatial.distance import pdist, euclidean
	from scipy.cluster.hierarchy import linkage, fcluster

	def dist(a, b):
		if list(a) == list(b):
			return 0.0
		degree_dist = (np.sum(a) - np.sum(b))**2 * 10
		prob_k = np.max(degree_distribution[int(np.sum(a))], degree_distribution[int(np.sum(b))])
		eucl_distance = euclidean(a,b)
		rel_eucl_distance = eucl_distance/np.sqrt(np.sum(a+b))
		return degree_dist + rel_eucl_distance * prob_k

	odes = list(m_vectors)
	Xd = pdist(odes, dist)
	Z = linkage(Xd, 'complete')
	indicators = fcluster(Z, cluster_number, criterion='maxclust')
	results = dict()
	for i in range(len(odes)):
		m = odes[i]
		results[m] = indicators[i]

	return results


def cluster_alt(degree_distribution, cluster_number, m_vectors):
	from scipy.spatial.distance import pdist, euclidean
	from scipy.cluster.hierarchy import linkage, fcluster

	def dist(a, b):
		if list(a) == list(b):
			return 0.0
		degree_dist = (np.sum(a) - np.sum(b))**2 * 10
		number_of_vectors_a = elemsin_k_vec_with_sum_m(len(a), np.sum(a))
		number_of_vectors_b = elemsin_k_vec_with_sum_m(len(b), np.sum(b))
		number_of_vectors = max(number_of_vectors_a, number_of_vectors_b)
		prob_mass_k_a = degree_distribution[int(np.sum(a))]
		prob_mass_k_b = degree_distribution[int(np.sum(b))]
		prob_mass_k = max(prob_mass_k_a, prob_mass_k_b )
		mass = prob_mass_k / number_of_vectors
		eucl_distance = euclidean(a,b)
		rel_eucl_distance = eucl_distance/np.sqrt(np.sum(a+b))
		return mass * degree_dist + mass * rel_eucl_distance

	odes = list(m_vectors)
	Xd = pdist(odes, dist)
	Z = linkage(Xd, 'complete')
	indicators = fcluster(Z, cluster_number, criterion='maxclust')
	results = dict()
	for i in range(len(odes)):
		m = odes[i]
		results[m] = indicators[i]

	return results

if __name__ == "__main__":
	degree_distribution = [0.1,0.1,0.1,0.1,0.6]
	cluster_number = 7
	m_vectors = list()
	for k in range(5):
		m_vectors += list(m_k_of(k,2))

	for cluster_method in [cluster, cluster_alt]:
		r = cluster_method(degree_distribution, cluster_number, m_vectors)
		print(r)
		inv_map = {}
		for k, v in r.items():
			inv_map[v] = inv_map.get(v, [])
			inv_map[v].append(k)
		print(inv_map)
	degree_distribution = [0.1,0.1,0.1,0.1,0.6]
	cluster_number = 7
	m_vectors = list()
	for k in range(15):
		m_vectors += list(m_k_of(k,2))
	print(cluster2(None, None, m_vectors))

def cluster1(degree_distribution, cluster_number, m_vectors):
	from scipy.spatial.distance import pdist, euclidean
	from scipy.cluster.hierarchy import linkage, fcluster
	odes = list(m_vectors)
	Xd = pdist(odes, lambda x,y: euclidean(x,y)*1.0/(np.sum(x+y)))
	Z = linkage(Xd, 'complete')
	indicators = fcluster(Z, cluster_number, criterion='maxclust')
	return indicators_to_dict(indicators,odes)

def cluster_direction(degree_distribution, cluster_number, m_vectors):
	from scipy.spatial.distance import pdist, euclidean
	from scipy.cluster.hierarchy import linkage, fcluster
	odes = list(m_vectors)
	def norma(v):
		return v if np.sum(v) == 0.0 else create_normalized_np(v)
	Xd = pdist(odes, lambda x,y: euclidean(norma(x), norma(y))+0.1*euclidean(x,y))
	Z = linkage(Xd, 'complete')
	indicators = fcluster(Z, cluster_number, criterion='maxclust')
	return indicators_to_dict(indicators,odes)


def cluster2(degree_distribution, cluster_number, m_vectors):
	logger.warn('This method ignores cluster_number and degree_distribution.')
	results = dict()
	for m in m_vectors:
		m_new = ''
		k = int(np.sum(m))
		for v in m:
			if k <= 8:
				m_new += '_'+str(v)
			else:
				m_new += '_'+str(int(v/3))
		results[m] = '{}{}'.format(k,m_new)
	return results

def cluster3(degree_distribution, cluster_number, m_vectors):
	logger.warn('This method ignores cluster_number and degree_distribution.')
	results = dict()
	for m in m_vectors:
		k = int(np.sum(m))
		if k == 0:
			k = 0.1
		c_id = [str(int(v/k*10)) for v in m]
		results[m] = '_'.join(c_id)
	return results

def cluster_vec(degree_distribution, cluster_number, m_vectors):
	from scipy.spatial.distance import euclidean
	logger.warn('This method ignores cluster_number and degree_distribution.')
	base = m_k_of(max(5,int(cluster_number)), len(m_vectors[0])) #arbitrary before 10
	#base = cluster_number
	base = [create_normalized_np(v, True) for v in base]
	results = dict()
	for m in m_vectors:
		k = int(np.sum(m))
		if k == 0: # values does not matter for 0-vector bu cannot be 0
			k = 0.1
		m_normal = [v/k for v in m]
		distances = [euclidean(b_v, m_normal) for b_v in base]
		closest_base = distances.index(min(distances))
		results[m] = '{}_{}'.format(k, closest_base)
	return results

def cluster_vec2(degree_distribution, cluster_number, m_vectors):
	from scipy.spatial.distance import euclidean
	logger.warn('This method ignores cluster_number and degree_distribution.')
	base = m_k_of(max(5,int(cluster_number)), len(m_vectors[0])) #arbitrary before 10
	#base = cluster_number
	base = [create_normalized_np(v, True) for v in base]
	results = dict()
	for m in m_vectors:
		k = int(np.sum(m))
		if k == 0: # values does not matter for 0-vector bu cannot be 0
			k = 0.1
		m_normal = [v/k for v in m]
		distances = [euclidean(b_v, m_normal) for b_v in base]
		closest_base = distances.index(min(distances))
		k_clustered = k if k <= 8 else int(k/3)*3  # vorher 5 und 2
		results[m] = '{}_{}'.format(k_clustered, closest_base)
	return results


def cluster_vec3(degree_distribution, cluster_number, m_vectors):
	from scipy.spatial.distance import euclidean
	logger.warn('This method ignores cluster_number and degree_distribution.')
	base = m_k_of(max(2,int(cluster_number)), len(m_vectors[0])) #arbitrary before 10
	#base = cluster_number
	base = [create_normalized_np(v, True) for v in base]
	results = dict()
	for m in m_vectors:
		k = int(np.sum(m))
		if k == 0: # values does not matter for 0-vector bu cannot be 0
			k = 0.1
		m_normal = [v/k for v in m]
		distances = [euclidean(b_v, m_normal) for b_v in base]
		closest_base = distances.index(min(distances))
		k_clustered = k if k <= 15 else int(k/3)*3  # vorher 5 und 2
		results[m] = '{}_{}'.format(k_clustered, closest_base)
	return results

def cluster_vec3s(degree_distribution, cluster_number, m_vectors):
	from scipy.spatial.distance import euclidean
	logger.warn('This method ignores cluster_number and degree_distribution.')
	base = m_k_of(max(2,int(cluster_number)), len(m_vectors[0])) #arbitrary before 10
	#base = cluster_number
	base = [create_normalized_np(v, True) for v in base]
	results = dict()
	for m in m_vectors:
		k = int(np.sum(m))
		if k == 0: # values does not matter for 0-vector bu cannot be 0
			k = 0.1
		m_normal = [v/k for v in m]
		distances = [euclidean(b_v, m_normal) for b_v in base]
		closest_base = distances.index(min(distances))
		k_clustered = k if k <= 4 else int(k/3)*3  # vorher 5 und 2
		results[m] = '{}_{}'.format(k_clustered, closest_base)
	return results

def cluster_vec4(degree_distribution, cluster_number, m_vectors):
	from scipy.spatial.distance import euclidean
	logger.warn('This method ignores cluster_number and degree_distribution.')
	base = m_k_of(max(2,int(cluster_number)), len(m_vectors[0])) #arbitrary before 10
	#base = cluster_number
	base = [create_normalized_np(v, True) for v in base]
	results = dict()
	for m in m_vectors:
		k = int(np.sum(m))
		if k == 0: # values does not matter for 0-vector bu cannot be 0
			k = 0.1
		m_normal = [v/k for v in m]
		distances = [euclidean(b_v, m_normal) for b_v in base]
		closest_base = distances.index(min(distances))
		k_clustered = k if k <= 12 else int(k/2)*2  # vorher 5 und 2
		results[m] = '{}_{}'.format(k_clustered, closest_base)
	return results

def cluster_vec5(degree_distribution, cluster_number, m_vectors):
	from scipy.spatial.distance import euclidean
	logger.warn('This method ignores cluster_number and degree_distribution.')
	base = m_k_of(10, len(m_vectors[0])) #arbitrary before 10
	#base = cluster_number
	base = [create_normalized_np(v, True) for v in base]
	results = dict()
	for m in m_vectors:
		k = int(np.sum(m))
		if k == 0: # values does not matter for 0-vector bu cannot be 0
			k = 0.1
		m_normal = [v/k for v in m]
		distances = [euclidean(b_v, m_normal) for b_v in base]
		closest_base = distances.index(min(distances))
		k_clustered = k if k <= 7 else int(k/5)*2  # vorher 5 und 2
		results[m] = '{}_{}'.format(k_clustered, closest_base)
	return results

def cluster_subspace(model):
	from scipy.spatial.distance import euclidean
	logger.warn('This method ignores cluster_number and degree_distribution.')
	merge = int(model.get('merge', 5))
	subspace = int(model.get('subspace', 7))
	start_clustering = int(model.get('start_clustering', 13))
	m_vectors = model['neighborhood']
	base = m_k_of(subspace, len(m_vectors[0])) #arbitrary before 10
	#base = cluster_number
	base = [create_normalized_np(v, True) for v in base]
	results = dict()
	for m in m_vectors:
		k = int(np.sum(m))
		if k == 0: # values does not matter for 0-vector bu cannot be 0
			k = 0.1
		m_normal = [v/k for v in m]
		distances = [euclidean(b_v, m_normal) for b_v in base]
		closest_base = distances.index(min(distances))
		k_clustered = k if k <= start_clustering else int(k/merge)*merge  # vorher 5 und 2
		results[m] = '{}_{}'.format(k_clustered, closest_base)
	return results

def cluster_subspace5(model):
	from scipy.spatial.distance import euclidean
	logger.warn('This method ignores cluster_number and degree_distribution.')
	m_vectors = model['neighborhood']
	base = m_k_of(5, len(m_vectors[0])) #arbitrary before 10
	base = [create_normalized_np(v, True) for v in base]
	degree_cluster = cluster_degrees(model)
	results = dict()
	for m in m_vectors:
		k = int(np.sum(m))
		if k == 0: # values does not matter for 0-vector bu cannot be 0
			k = 1
		m_normal = [v/k for v in m]
		distances = [euclidean(b_v, m_normal) for b_v in base]
		closest_base = distances.index(min(distances))
		results[m] = '{}_{}'.format(degree_cluster[k], closest_base)
	return results

def cluster_subspace7(model):
	from scipy.spatial.distance import euclidean
	logger.warn('This method ignores cluster_number and degree_distribution.')
	m_vectors = model['neighborhood']
	base = m_k_of(7, len(m_vectors[0])) #arbitrary before 10
	base = [create_normalized_np(v, True) for v in base]
	degree_cluster = cluster_degrees(model)
	results = dict()
	for m in m_vectors:
		k = int(np.sum(m))
		if k == 0: # values does not matter for 0-vector bu cannot be 0
			k = 1
		m_normal = [v/k for v in m]
		distances = [euclidean(b_v, m_normal) for b_v in base]
		closest_base = distances.index(min(distances))
		results[m] = '{}_{}'.format(degree_cluster[k], closest_base)
	return results

def cluster_subspaceS(model):
	from scipy.spatial.distance import euclidean
	logger.warn('This method ignores cluster_number and degree_distribution.')
	m_vectors = model['neighborhood']
	base = m_k_of(int(model.get('subspace', 30)), len(m_vectors[0])) #arbitrary before 10
	base = [create_normalized_np(v, True) for v in base]
	degree_cluster = cluster_degrees(model)
	results = dict()
	for m in m_vectors:
		k = int(np.sum(m))
		if k == 0: # values does not matter for 0-vector bu cannot be 0
			k = 1
		m_normal = [v/k for v in m]
		distances = [euclidean(b_v, m_normal) for b_v in base]
		closest_base = distances.index(min(distances))
		results[m] = '{}_{}'.format(degree_cluster[k], closest_base)
	return results

def cluster_subspaceX(model):
	from scipy.spatial.distance import euclidean, cosine
	logger.warn('This method ignores cluster_number.')
	m_vectors = model['neighborhood']
	base = m_k_of(int(model.get('subspace', model['bin_num'])), len(m_vectors[0])) #arbitrary before 10
	base = [create_normalized_np(v, True) for v in base]
	degree_cluster = cluster_degrees(model)
	results = dict()
	for m in m_vectors:
		k = int(np.sum(m)+0.0000001)
		m_normal = [v/k for v in m] if k > 0 else [0.001 for v in m]
		m1 = m if k>0 else np.ones(len(m))
		#distances = [euclidean(b_v, m_normal) for b_v in base]
		distances = [cosine(b_v, m1) for b_v in base] # todo
		closest_base = distances.index(min(distances))
		results[m] = '{}_{}'.format(degree_cluster[k], closest_base)
	return results

def cluster_subspaceXX(model):
	model['subspace'] = int(model['bin_num']*0.5) # gamma = bin_num / k
												 # gamma * k = bin_num
												 # k = bin_num / gamma
	return cluster_subspaceX(model)

def cluster_subspaceXY(model):
	model['subspace'] = int(model['bin_num']*1.0)
	return cluster_subspaceX(model)

def cluster_subspaceXZ(model):
	model['subspace'] = int(model['bin_num']*2.0)
	return cluster_subspaceX(model)

def cluster_fixedSubSpace9(model):
	model['subspace'] = 9
	return cluster_subspaceX(model)
def cluster_fixedSubSpace10(model):
	model['subspace'] = 10
	return cluster_subspaceX(model)
def cluster_fixedSubSpace20(model):
	model['subspace'] = 20
	return cluster_subspaceX(model)

def cluster_fixedBinNum5(model):
	moodel['subspace'] = int(model['bin_num']*1.0)
	model['bin_num'] = 5
	return cluster_subspaceX(model)
def cluster_fixedBinNum10(model):
	moodel['subspace'] = int(model['bin_num']*1.0)
	model['bin_num'] = 10
	return cluster_subspaceX(model)
def cluster_fixedBinNum20(model):
	moodel['subspace'] = int(model['bin_num']*1.0)
	model['bin_num'] = 20
	return cluster_subspaceX(model)

def plot_clustering(cluster_results, outpath):
	global created_images
	import matplotlib as mpl
	import matplotlib.pyplot as plt
	import seaborn as sns
	sns.set_style("white")
	mpl.rc('xtick', labelsize=14)
	mpl.rc('ytick', labelsize=14)
	import numpy as np
	plt.clf()
	ax = plt.subplot(111, aspect='equal')
	cluster_ids = sorted(list(set(cluster_results.values())))
	vectors = cluster_results.keys()
	# for more than 2 dimensions: only consider slice
	x = [v[0] for v in vectors if np.sum(v) == v[0]+v[1]]
	y = [v[1] for v in vectors if np.sum(v) == v[0]+v[1]]
	ax.set_ylim([-.1, max(x+y)*1.05])
	ax.set_xlim([-.1, max(x+y)*1.05])
	colors_per_cluster = np.random.rand(len(cluster_ids))
	#colors_per_cluster = np.linspace(0,1,len(cluster_ids))
	colors = list()
	for v in vectors:
		c = cluster_results[v]
		c_pos = cluster_ids.index(c)
		colors.append(colors_per_cluster[c_pos])
	colors = [plt.get_cmap('hsv')(c) for c in colors ]
	#colors = [str(color) for color in colors]
	#plt.show()
	plt.scatter(x, y, c=colors, alpha=0.8, s = 10)
	plt.savefig(outpath, format='pdf', bbox_inches='tight')


#------------------------------------------------------
# Heuristics for Degree Clustering
#------------------------------------------------------

def cluster_degrees(model):
	import DegreeClusterEngine as dce
	#clustering = dce.hierarchical2d_woblist(model)
	clustering = dce.loss_clustering(model)
	return clustering
	#return dce.greedy(model)
	#return {k: int(k) for k in range(model['k_max']+1)}
