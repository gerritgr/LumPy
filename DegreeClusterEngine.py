from utilities import *
import numpy as np
import collections

#------------------------------------------------------
# Config
#------------------------------------------------------

RECORD_LOSS = False

#------------------------------------------------------
# Binning tools
#------------------------------------------------------

def get_allowed_heuristics():
	return 'scipy_hierarchical,kmeans,greedy,uniform,hierarchical2d,hierarchical2d_woblist'.split(',')

def bin_function(b, function, binning, distribution):
	if len(binning) != len(distribution):
		raise ValueError('Binning does not match distribution.')
	partial_bin = list()
	partial_bin_index = list()
	for i in range(len(binning)):
		if binning[i] == b:
			partial_bin_index.append(i)
			partial_bin.append(distribution[i])
	partition = np.sum(partial_bin)
	if isclose(partition, 0.0):
		raise ValueError('Unkown error during computation of bin statistics: '+str(b)+' '+str(binning))
	partial_bin = [float(v)/partition for v in partial_bin]
	result = 0.0
	for i in range(len(partial_bin)):
		result += partial_bin[i]*function(partial_bin_index[i])
	return result

def bin_statistics_old(function, binning, distribution):
	stats = dict()
	for b in range(len(set(binning))):
		v = bin_function(b, function, binning, distribution)
		stats[b] = v
	return stats

def bin_statistics(function, binning, distribution):
	bin_sums = dict()
	stats_unscaled = dict()
	stats = dict()
	for i in range(len(distribution)):
		b = binning[i]
		bin_sums[b] = distribution[i] + bin_sums.get(b, 0.0)
	for i in range(len(distribution)):
		v = function(i)
		b = binning[i]
		stats_unscaled[b] = distribution[i] * v + stats_unscaled.get(b, 0.0)
	stats = {b:(v/bin_sums[b]) for b,v in stats_unscaled.items()}
	return stats

def binning_to_str(binning):
	if type(binning) is dict:
		binning = [binning.get(i,0.0) for i in range(int(np.max(list(binning.keys()))+1.5))]
	binning = np.array(binning)
	bincount = hist(binning)
	if len(bincount.keys()) > 0.9*len(binning):
		return to_str(binning)

	bin_str = str(binning.tolist())
	s = ''
	for b in sorted(bincount.keys()):
		s += '+[{b}]*{l}'.format(b = b, l = int(bincount[b]))
	s = s[1:]
	# s_list = eval(s)
	# if s_list != list(binning):
	# 	for i in range(len(s_list)):
	# 		if s_list[i] != binning[i]:
	# 			logger.error('error here', s_list[i], binning[i])
	# 	logger.error(list(binning))
	# 	logger.error(s)
	# 	raise ValueError('Something went wrong.')
	if len(s) < len(bin_str):
		return s
	return bin_str

def binning_error(binning, degree_distribution):
	means = bin_statistics(lambda x: x, binning, degree_distribution)
	errors = np.zeros(len(binning))
	for i in range(len(errors)):
		errors[i] = np.abs(means[binning[i]] - i)
	return np.mean(errors)


def call_bin_heuristic(model):
	v = model['degree_distribution']
	bin_num = model['bin_num']
	heuristic = model['heuristic']

	logger.info('compute binning for {} bins'.format(bin_num))

	if bin_num <= 0:
		raise ValueError('The number of bins has to be positive.')

	if bin_num > len([value for value in v if not isclose(value, 0.0)]):
		raise ValueError('The number of bins must not be larger than the number of degrees with positive probability.')

	if isinstance(heuristic, str):
		heuristic = heuristic.strip()
		#print('heuristic', heuristic)
		if not heuristic in get_allowed_heuristics():
			raise ValueError('unknown binning heuristic')
		heuristic = globals()[heuristic]
	else:
		if not isinstance(heuristic, collections.Callable):
			raise ValueError('Binning heuristic is not callable.')

	# normalize
	# partition = np.sum(v)
	# v_new = [float(e)/partition for e in v]
	# v = v_new
	v = create_normalized_np(v)

	if bin_num == 1:
		logger.info('use a single bin')
		return [0]*len(v)
	if bin_num == len([value for value in v if not isclose(value, 0.0)]):
		# there is only one binning possibility, cdfmean finds it.
		logger.info('only one binning possibility')
		indicator_list = np.zeros(len(v), dtype=np.int)
		for i in range(len(v)):
			if i == 0:
				indicator_list[i] = 0
			else:
				if isclose(v[i], 0.0):
					indicator_list[i] = indicator_list[i-1]
				elif indicator_list[i-1] == 0 and isclose(v[i-1], 0.0):
					indicator_list[i] = 0
				else:
					indicator_list[i] = indicator_list[i-1]+1
		return indicator_list
	logger.info('start binning heuristic')
	binning = heuristic(model)
	return binning


def bin_initial_vector_dbmf(model):
	initial_vector = np.array(model['initial_vector'])
	state_num = len(model['states'])
	degree_num = model['k_max']+1
	degree_distribution = model['degree_distribution']
	bin_num = model['bin_num']
	bins = model['bins']

	try:
		v = initial_vector.reshape(state_num, degree_num)
	except:
		print(len(initial_vector))
		print('state num', state_num)
		print('degree num', degree_num)
		raise ValueError('Lengh of initial_vector is incorrect.')
	partition = np.sum(v, axis=0)
	if np.count_nonzero(partition) != partition.size:
		raise ValueError('Initial distribution cannot be zero for any k.')
	v = v/partition

	binned_initial = np.zeros((state_num, bin_num))
	for state_i in range(state_num):
		for k in range(len(degree_distribution)):
			prob_mass = degree_distribution[k]
			bin_i = bins[k]
			binned_initial[state_i, bin_i] += prob_mass * v[state_i, k]

	partition = np.sum(binned_initial, axis=0)
	binned_initial = binned_initial/partition
	return binned_initial.flatten()


def bin_initial_vector_pa(model):
	initial_vector = np.array(model['initial_vector'])
	state_num = len(model['states'])
	degree_num = model['k_max']+1
	degree_distribution = model['degree_distribution']
	bin_num = model['bin_num']
	bins = model['bins']

	try:
		v_states_flat = initial_vector[:state_num*degree_num]
		v_betas_flat = initial_vector[state_num*degree_num:]

		v_states = v_states_flat.reshape(state_num, degree_num)
		v_betas = v_betas_flat.reshape(state_num, state_num, degree_num)
	except:
		print(len(initial_vector))
		print('state num', state_num)
		print('degree num', degree_num)
		raise ValueError('Lengh of initial_vector is incorrect.')

	v_states = v_states/np.sum(v_states, axis = 0)
	for i in range(state_num):
		v_betas[i] = v_betas[i]/(np.sum(v_betas,1)[i])

	binned_states = np.zeros((state_num, bin_num))
	binned_betas = np.zeros((state_num, state_num, bin_num))

	for state_i in range(state_num):
		for k in range(len(degree_distribution)):
			prob_mass = degree_distribution[k]
			bin_i = bins[k]
			binned_states[state_i, bin_i] += prob_mass * v_states[state_i, k]

	for state_i in range(state_num):
		for state_j in range(state_num):
			for k in range(len(degree_distribution)):
				prob_mass = degree_distribution[k]
				bin_i = bins[k]
				binned_betas[state_i, state_j, bin_i] += prob_mass * v_betas[state_i, state_j, k]

	binned_states = binned_states/np.sum(binned_states, axis = 0)
	for i in range(state_num):
		binned_betas[i] = binned_betas[i]/(np.sum(binned_betas,1)[i])

	v_result = np.append(binned_states.flatten(),(binned_betas.flatten()))
	return v_result

#------------------------------------------------------
# Binning heuristics
#------------------------------------------------------

def loss_clustering(model):
	alpha = float(model.get('hierarchical_coefficient', 0.9))
	#alpha = float(model.get('hierarchical_coefficient', 0.7))
	def surrogate_loss(cluster):
		degree_list, prob_list = cluster
		prob_mass = np.sum(prob_list)
		if isclose(prob_mass, 0.0):
			return 0.0
		mean_degree = 0.0
		for i, degree in enumerate(degree_list):
			mean_degree += degree * (prob_list[i]/prob_mass)
		sd = 0.0
		for i, degree in enumerate(degree_list):
			sd += np.abs(degree - mean_degree)**2 * (prob_list[i]/prob_mass)
		if degree_list == [0]:
			loss = prob_mass ** (2)
		else:
			loss = prob_mass ** (2) * alpha + (1-alpha) * np.var(degree_list)/np.mean(degree_list)
		assert(not np.isnan(loss))
		return loss

	bin_num = model['bin_num']
	degree_distribution = model['degree_distribution']
	clustering = [([i],[degree_distribution[i]]) for i in range(len(degree_distribution))]
	if RECORD_LOSS:
		model['loss_list'] = list()
		model['loss_list'].append(np.sum([surrogate_loss(c) for c in clustering]))
	final_clustering = list(clustering)
	while len(clustering) > 1:
		best_costs = 1000**2
		best_i = -1
		best_joint = None
		for i in range(len(clustering)-1):
			c1 = clustering[i]
			c2 = clustering[i+1]
			costs_c1 = surrogate_loss(c1)
			costs_c2 = surrogate_loss(c2)
			cjoin = (c1[0]+c2[0],c1[1]+c2[1])
			costs_cjoin = surrogate_loss(cjoin)
			join_costs = costs_cjoin - (costs_c1 + costs_c2)
			assert(costs_cjoin >= (costs_c1 + costs_c2))
			if join_costs<best_costs:
				best_costs=join_costs
				best_i = i
				best_joint = cjoin
		clustering[best_i] = best_joint
		del clustering[best_i+1]
		if RECORD_LOSS:
			model['loss_list'].append(np.sum([surrogate_loss(c) for c in clustering]))
			if len(clustering) == bin_num:
				model['effective_loss'] = model['loss_list'][-1]
		if len(clustering) == bin_num:
			final_clustering = list(clustering)
	indicators = list()
	for i, cluster in enumerate(final_clustering):
		indicators += [i]*len(cluster[0])

	if RECORD_LOSS:
		logger.info('surrogate loss is:\t{}'.format(model['loss_list']))
	return indicators

# def test_loss():
# 	model = {'bin_num':4, 'degree_distribution' : [1/20.0 for i in range(20)]}
# 	print (loss_clustering(model))
#
# test_loss()



def greedy_naive(model):
	v = model['degree_distribution']
	bin_num = model['bin_num']

	v = create_normalized_np(v)
	logger.info('start greedy heuristic')
	bins_remaining = bin_num
	prob_mass_remaining = 1.0
	prob_mass_current_bin = 0.0
	cluster_indicators = [-1] * len(v)
	prob_mass_per_bin = 1.0/bins_remaining
	for i in range(len(v)):
		#print 'prob_mass_per_bin', prob_mass_per_bin, 'prob_mass_current_bin', prob_mass_current_bin, 'bins_remaining', bins_remaining, 'v_i', v[i]
		if prob_mass_current_bin + v[i] > prob_mass_per_bin and bins_remaining > 0 and i != 0 and prob_mass_current_bin != 0.0:
			cluster_indicators[i] = cluster_indicators[i-1] + 1
			bins_remaining -= 1
			#prob_mass_per_bin = prob_mass_remaining/bins_remaining if bins_remaining > 0 else 2.0
			prob_mass_remaining -= v[i]
			prob_mass_current_bin = v[i]
		else:
			cluster_indicators[i] = cluster_indicators[i-1] if i>0 else 0
			prob_mass_current_bin += v[i]
			prob_mass_remaining -= v[i]
			if i == 0:
				bins_remaining -= 1

	if cluster_indicators[-1] != bin_num-1:
		raise ValueError('could not cluster distribution: \n'+str(cluster_indicators[:30])+'\n'+str(v))

	logger.info('end greedy heuristic')
	return cluster_indicators

def greedy(model):
	v = model['degree_distribution']
	bin_num = model['bin_num']

	v = create_normalized_np(v)
	avg_probmass = 1.0/bin_num
	for i in range(20): #20 being arbitrary here
		v = np.minimum(v, [avg_probmass] * len(v))
		v = create_normalized_np(v)
	for i in range(20):
		try:
			model['degree_distribution'] = v
			return greedy_naive(model)
		except:
			v = np.power(v, 0.9) #equalizes values
	raise ValueError('could not cluster distribution: \n'+str(cluster_indicators[:30])+'\n'+str(v))

def uniform(model):
	# not exact
	v = model['degree_distribution']
	bin_num = model['bin_num']

	positive_count = len([prob for prob in v if not isclose(prob, 0.0)])
	f = lambda x: 0.0 if isclose(x, 0.0) else 1.0/positive_count
	v = [f(prob) for prob in v]
	model['degree_distribution'] = v
	return greedy(model)

def kmeans_naive(model):
	from scipy.cluster.vq import kmeans2
	bin_num = model['bin_num']
	degree_distribution = model['degree_distribution']
	distance = create_distance_vector(degree_distribution)
	clustering = kmeans2(distance, bin_num)[1]
	return sort_clusterids(clustering)

def kmeans(model):
	for _ in range(100):
		cluster = kmeans_naive(model)
		# this might happen if some cluster is empty
		if len(set(cluster)) == model['bin_num']:
			return cluster
	logger.error('Could not find binning with kmeans. Use greedy heuristic.')
	return greedy(model)

def create_distance_vector(degree_distribution):
	cdf = np.zeros(len(degree_distribution))
	distance = np.zeros(len(degree_distribution))

	for i in range(len(degree_distribution)):
		if i == 0:
			cdf[i] = degree_distribution[i]
		else:
			cdf[i] = cdf[i-1] + degree_distribution[i]
	for i in range(len(degree_distribution)):
		if i == 0:
			distance[i] = degree_distribution[i]/2.0
		else:
			distance[i] = degree_distribution[i]/2.0 + cdf[i-1]

	# handle zero probabilities
	for i in range(len(degree_distribution)):
		p = degree_distribution[i]
		if not isclose(p, 0.0):
			continue
		for offset in range(len(degree_distribution)):
			try:
				if not isclose(0.0, degree_distribution[i+offset]):
					distance[i] = distance[i+offset]
					break
			except:
				pass
			try:
				if not isclose(0.0, degree_distribution[i-offset]):
					distance[i] = distance[i-offset]
					break
			except:
				pass
	#print('new distance distribution ',distance)
	distance = np.multiply(distance, 2)
	return distance

def sort_clusterids(clustering):
	centroid_id = list()
	for k in clustering:
		if k not in centroid_id:
			centroid_id.append(k)
	centroid_replace = {i: centroid_id.index(i) for i in centroid_id}
	clustering = [centroid_replace[i] for i in clustering]
	return clustering

def scipy_hierarchical(model, method = None, metric = None):
	from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
	from scipy.spatial.distance import pdist
	if 'scipy_hierarchy_method' in model and 'str' in str(type(model['scipy_hierarchy_method'])):
		method = model['scipy_hierarchy_method']
	if 'scipy_hierarchy_metric' in model and 'str' in str(type(model['scipy_hierarchy_metric'])):
		metric = model['scipy_hierarchy_metric']
	if method is None:
		method = 'ward'
	if metric is None:
		metric = 'euclidean'

	distance = create_distance_vector(model['degree_distribution'])
	Y = pdist([[d] for d in distance], metric)
	Z = linkage(Y, method, metric)
	indicators = fcluster(Z, model['bin_num'], criterion='maxclust')
	if len(set(indicators)) != model['bin_num']:
		logger.error('Number of indicators does not match number of bins')
		logger.error('indicators: ' +str(indicators))
		logger.error('Number of bins is: '+str(model['bin_num']))
		raise ValueError('Somehow the scipy clustering results are wrong.')
	indicators = sort_clusterids(indicators)
	return indicators

hpmfmean2d_results = dict()
join_history = list()
def hierarchical2d_woblist_old(model):
	global hpmfmean2d_results

	degree_distribution = model['degree_distribution']
	bin_num = model['bin_num']

	if bin_num <= 0:
		raise ValueError('The number of bins has to be positive.')
	if bin_num > len([value for value in degree_distribution if not isclose(value, 0.0)]):
		raise ValueError('The number of bins must not be larger than the number of degrees with positive probability.')

	# see clustering is already done
	degree_tuple = str(model['modeltext'])
	if degree_tuple in hpmfmean2d_results:
		join_history = hpmfmean2d_results[degree_tuple]
	else:
		join_history = list()

	def distance_function(c1, c2):
		w = 0.1 if 'hierarchical_coefficient' not in model else float(model['hierarchical_coefficient'])
		prob_mass1, mean_degree1, _, d1to2 = c1
		prob_mass2, mean_degree2, _, d2to3 = c2
		if not np.isnan(d1to2):
			return d1to2
		if isclose(prob_mass1, 0.0) or isclose(prob_mass2, 0.0):
		 	c1[3] = 0.0
		 	return 0.0
		relative_degree_distance = np.abs(mean_degree1 - mean_degree2)/(0.5*(mean_degree1 + mean_degree2))
		entropy_distance = (-np.log(c1[0])*c1[0]-np.log(c2[0])*c2[0])  -  (-np.log(c1[0]+c2[0])*(c1[0]+c2[0]))
		prob_distance = prob_mass1 + prob_mass2
		relative_degree_distance = np.abs(mean_degree1 - mean_degree2)/(0.5*(mean_degree1 + mean_degree2))
		distance = prob_distance + w * relative_degree_distance
		#distance = np.sqrt(entropy_distance) * (relative_degree_distance) # * np.sqrt(prob_mass1)
		c1[3] = distance
		return distance

	def joint_clusters(c1, c2):
		''' Caution: joins lists in c1 due to performance reasons '''
		prob_mass1, mean_degree1, degree_list1, d1to2 = c1
		prob_mass2, mean_degree2, degree_list2, d2to3 = c2

		prob_mass_new = prob_mass1 + prob_mass2
		num_elems = float(len(degree_list1) + len(degree_list2))
		mean_degreee_new = len(degree_list1)/num_elems * mean_degree1 + len(degree_list2)/num_elems * mean_degree2
		degree_list1 += degree_list2 #faster than creation of new list
		return [prob_mass_new, mean_degreee_new, degree_list1, np.nan]

	clustering = [[degree_distribution[i], i, [i], np.nan] for i in range(len(degree_distribution))]

	if len(clustering) < 2:
		raise ValueError('Could not cluster distribution, too little binss')

	step = -1
	while len(clustering) > bin_num:
		step += 1

		# find left_i
		if step < len(join_history):
			left_i = join_history[step]
		else:
			min_distance = distance_function(clustering[0], clustering[1])
			left_i = 0
			for i in range(len(clustering)-1):
				distance_i = distance_function(clustering[i], clustering[i+1])
				if distance_i < min_distance:
					left_i = i
					min_distance = distance_i
			join_history.append(left_i)

		# join clusters
		new_cluster = joint_clusters(clustering[left_i], clustering[left_i+1])
		clustering = clustering[:left_i] + [new_cluster] + clustering[left_i+2:]
		if left_i > 0:
			clustering[left_i-1][3] = np.nan

	import itertools
	indicator_list = list()
	for i in range(len(clustering)):
		elem_num = len(clustering[i][2])
		indicator_list += [i] * elem_num

	hpmfmean2d_results[degree_tuple] = join_history
	return indicator_list

hierarchical2dblist_results = dict()
def load_results(model, c):
	global hierarchical2dblist_results
	if model['modeltext'] not in hierarchical2dblist_results:
		hierarchical2dblist_results[model['modeltext']] = list()
		return
	join_list = hierarchical2dblist_results[model['modeltext']]
	for i in join_list:
		#print('do clustering')
		if len(c) == model['bin_num']:
			return
		next_c = c[i][4]
		c[i][0] += c[next_c][0]
		c[i][1] = min(c[i][1], c[next_c][1])
		c[i][2] = max(c[i][2], c[next_c][2])
		c[i][3] = np.nan if np.isnan(c[i][3]) or np.isnan(c[next_c][3]) else min(c[i][3], c[next_c][3])
		c[i][4] = np.nan if np.isnan(c[i][4]) or np.isnan(c[next_c][4]) else max(c[i][4], c[next_c][4])
		del c[next_c]

def hierarchical2d(model):
	global hierarchical2dblist_results
	try:
		# use pip install blist
		from blist import sortedlist
	except:
		logger.warning('\nCould not import blist, use "pip install blist" to install the blist package for faster binning.\n')
		return hierarchical2d_woblist(model)

	degree_distribution = model['degree_distribution']
	bin_num = model['bin_num']
	distance = None

	def distance2d(c1, c2, model):
		weight = 0.1 if 'hierarchical_coefficient' not in model else model['hierarchical_coefficient']
		prob_mass1, min_k1, max_k1, prev1, next1, d1to2 = c1
		prob_mass2, min_k2, max_k2, prev2, next2, d2to3 = c2
		if isclose(0.0, prob_mass1) or isclose(0.0, prob_mass2):
			return 0.0

		#from scipy.stats import entropy
		#return entropy([prob_mass1, prob_mass2, 1.0-prob_mass1-prob_mass2])

		probability_distance = prob_mass1 + prob_mass2

		mean_degree1 = (min_k1 + max_k1)/2.0
		mean_degree2 = (min_k2 + max_k2)/2.0
		relative_distance = np.abs(mean_degree1 - mean_degree2)/(0.5*(mean_degree1 + mean_degree2))
		d = probability_distance + weight * relative_distance
		return d

	def join_cluster(c1, c2):
		logger.debug('join Clusters {} {}', c1, c2)
		c1[0] += c2[0]
		c1[1] = min(c1[1], c2[1])
		c1[2] = max(c1[2], c2[2])
		if np.isnan(c1[3]) or np.isnan(c2[3]):
			c1[3] = np.nan
		else:
			c1[3] = min(c1[3], c2[3])
		if np.isnan(c1[4]) or np.isnan(c2[4]):
			c1[4] = np.nan
		else:
			c1[4] = max(c1[4], c2[4])

	if 'hierarchical_distance' in model:
		distance = eval(model['hierarchical_distance'])
	else:
		distance = distance2d

	# id : probability mass, min k, max k, prev, next, distance to next
	clustering = {i: [degree_distribution[i], i, i, i-1, i+1, np.nan] for i in range(len(degree_distribution))}
	clustering[0][3] = np.nan
	clustering[len(degree_distribution)-1][4] = np.nan

	load_results(model, clustering)
	join_list = hierarchical2dblist_results[model['modeltext']]

	distances = list()
	for i in clustering:
		if i == max(clustering.keys()):
			continue
		c1 = clustering[i]
		c2 = clustering[c1[4]]
		d = distance(c1, c2, model)
		distances.append((d, i))
		c1[5] = d

	distances = sortedlist(distances, key = lambda x: x[0])
	#print(distances)

	while len(clustering) > bin_num:
		_, c_id1  = distances[0]

		join_list.append(c_id1)

		c1 = clustering[c_id1]
		c_id0 = c1[3]
		c0 = clustering.get(c_id0, np.nan)
		c_id2 = c1[4]
		c2 = clustering[c_id2]
		c_id3 = c2[4]
		c3 = clustering.get(c_id3, np.nan)

		#test integrity
		#print(clustering[c_id1][5],distance(c1,c2,model))
		#print('join cluster ', c_id1, ' beeing ', c1, c2)

		#join clusters
		join_cluster(c1, c2)
		del clustering[c_id2]
		if not np.isnan(c2[5]):
			distances.remove((c2[5], c_id2))

		if 'list' in str(type(c0)):
			distances.remove((c0[5], c_id0))
			c0[5] = distance(c0, c1, model)
			distances.add((c0[5], c_id0))

		distances.remove((c1[5], c_id1))
		if 'list' in str(type(c3)):
			c1[5] = distance(c1, c3, model)
			distances.add((c1[5], c_id1))
			c3[3] = c_id1
		else:
			c1[5] = np.nan

	indicators = list()
	for i in sorted(clustering.keys()):
		indicator = indicators[-1]+1 if len(indicators) > 0 else 0
		cluster = clustering[i]
		indicators += [indicator] * (cluster[2]-cluster[1]+1)
	#print(indicators)
	return indicators


def hierarchical2d_woblist(model):
	degree_distribution = model['degree_distribution']
	bin_num = model['bin_num']
	distance = None

	def distance2d(c1, c2, model):
		weight = 0.1 if 'hierarchical_coefficient' not in model else model['hierarchical_coefficient']
		prob_mass1, min_k1, max_k1, prev1, next1, d1to2 = c1
		prob_mass2, min_k2, max_k2, prev2, next2, d2to3 = c2
		if isclose(0.0, prob_mass1) or isclose(0.0, prob_mass2):
			return 0.0

		#from scipy.stats import entropy
		#return entropy([prob_mass1, prob_mass2, 1.0-prob_mass1-prob_mass2])

		probability_distance = prob_mass1 + prob_mass2

		mean_degree1 = (min_k1 + max_k1)/2.0
		mean_degree2 = (min_k2 + max_k2)/2.0
		relative_distance = np.abs(mean_degree1 - mean_degree2)/(0.5*(mean_degree1 + mean_degree2))
		d = probability_distance + weight * relative_distance
		return d

	def join_cluster(c1, c2):
		logger.debug('join Clusters {} {}', c1, c2)
		c1[0] += c2[0]
		c1[1] = min(c1[1], c2[1])
		c1[2] = max(c1[2], c2[2])
		if np.isnan(c1[3]) or np.isnan(c2[3]):
			c1[3] = np.nan
		else:
			c1[3] = min(c1[3], c2[3])
		if np.isnan(c1[4]) or np.isnan(c2[4]):
			c1[4] = np.nan
		else:
			c1[4] = max(c1[4], c2[4])

	if 'hierarchical_distance' in model:
		distance = eval(model['hierarchical_distance'])
	else:
		distance = distance2d

	# id : probability mass, min k, max k, prev, next, distance to next
	clustering = {i: [degree_distribution[i], i, i, i-1, i+1, np.nan] for i in range(len(degree_distribution))}
	clustering[0][3] = np.nan
	clustering[len(degree_distribution)-1][4] = np.nan

	load_results(model, clustering)
	join_list = hierarchical2dblist_results[model['modeltext']]

	distances = list()
	for i in clustering:
		if i == max(clustering.keys()):
			continue
		c1 = clustering[i]
		c2 = clustering[c1[4]]
		d = distance(c1, c2, model)
		distances.append((d, i))
		c1[5] = d

	distances = sorted(distances, key = lambda x: x[0])
	#print(distances)

	while len(clustering) > bin_num:
		_, c_id1  = distances[0]

		join_list.append(c_id1)

		c1 = clustering[c_id1]
		c_id0 = c1[3]
		c0 = clustering.get(c_id0, np.nan)
		c_id2 = c1[4]
		c2 = clustering[c_id2]
		c_id3 = c2[4]
		c3 = clustering.get(c_id3, np.nan)

		#test integrity
		#print(clustering[c_id1][5],distance(c1,c2,model))
		#print('join cluster ', c_id1, ' beeing ', c1, c2)

		#join clusters
		join_cluster(c1, c2)
		del clustering[c_id2]
		if not np.isnan(c2[5]):
			distances.remove((c2[5], c_id2))

		if 'list' in str(type(c0)):
			distances.remove((c0[5], c_id0))
			c0[5] = distance(c0, c1, model)
			distances.append((c0[5], c_id0))

		distances.remove((c1[5], c_id1))
		if 'list' in str(type(c3)):
			c1[5] = distance(c1, c3, model)
			distances.append((c1[5], c_id1))
			c3[3] = c_id1
		else:
			c1[5] = np.nan

		distances = sorted(distances, key = lambda x: x[0])

	indicators = list()
	for i in sorted(clustering.keys()):
		indicator = indicators[-1]+1 if len(indicators) > 0 else 0
		cluster = clustering[i]
		indicators += [indicator] * (cluster[2]-cluster[1]+1)
	#print(indicators)
	return indicators


# logger.info('start')
# for binnum in [50]:
# 	v = range(20000)
# 	v = create_normalized_np(v)
# 	binning = hierarchical2d_woblist({'degree_distribution': v, 'bin_num': binnum, 'modeltext': 'xxx'})
# 	print(binning[:50])
# 	logger.info('done 1')
# 	hierarchical2dblist_results = dict()
# 	binning = hierarchical2d_woblist_old({'degree_distribution': v, 'bin_num': binnum, 'modeltext': 'xxx'})
# 	print(binning[:50])
# 	logger.info('done 2')
# 	binning = hierarchical2d({'degree_distribution': v, 'bin_num': binnum, 'modeltext': 'xxx'})
# 	print(binning[:50])
# 	logger.info('done 2')
