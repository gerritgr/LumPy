import os,sys
sys.path.append('PA_DBMF/')
from model_parser import read_model
import numpy as np
import random
from utilities import *
import pandas as pd
import matplotlib as mlp
import matplotlib.pyplot as plt
import time

#------------------------------------------------------
# Globals
#------------------------------------------------------

node_map = dict()

#------------------------------------------------------
# Create Graph
#------------------------------------------------------

def create_random_graph(number_nodes, degree_distribution, states, initial_distribution):
	import networkx as nx
	max_degree = len(degree_distribution)-1
	z = np.sum(degree_distribution)
	degree_distribution = [p/z for p in degree_distribution]
	while True:
		degee_sequence = [np.random.choice(range(max_degree+1), p=degree_distribution) for _ in range(number_nodes)]
		if np.sum(degee_sequence) % 2 == 0:
			break
	labels = {node:np.random.choice(states, p = initial_distribution) for node in range(number_nodes)}
	G=nx.configuration_model(degee_sequence)
	G=nx.Graph(G)
	G.remove_edges_from(G.selfloop_edges())
	graph = (G.edges(), labels)
	generate_nodemap(G.edges())
	return graph


def generate_nodemap(edges_new):
	global node_map
	node_map = dict()
	# new storage
	for i,j in edges_new:
		if i not in node_map:
			node_map[i] = list()
		if j not in node_map:
			node_map[j] = list()
		node_map[i].append(j)
		node_map[j].append(i)

def randomize_edges(edges):
	''' input of the form [(56, '_'), (920, '_'), (28, '_'), (664, '_'), (828, '_'), (394, '_'), (20, '_'), (590, '_')] '''

	while True:
		logger.info("shuffle edges...")
		edge_candidates = [e for e in edges] #copy
		random.shuffle(edge_candidates)
		edges_sorted = [edge_candidates.pop(0)]

		for _ in range(len(edge_candidates) + 1): #upper bound for operations if it works
			for i in range(len(edge_candidates)):
				edge_candidate = edge_candidates[i]
				if edge_candidate != edges_sorted[-1]:
					edges_sorted.append(edge_candidates.pop(i))
					break

		if len(edge_candidates) == 0:
			return edges_sorted
		else:
			logger.info("not successfull")
			print(edge_candidates)

def generate_network(model):

	degree_distribution = model['degree_distribution']
	degree_distribution[0] = 0.0
	s = np.sum(degree_distribution)
	degree_distribution = [prob/s for prob in degree_distribution]

	node_degrees = np.random.choice(list(range(len(degree_distribution))), model['number_of_nodes'], p=degree_distribution)

	#print 'node degrees', node_degrees

	edges = list()
	for i in range(len(node_degrees)):
		for j in range(node_degrees[i]):
			edges.append((i, "_"))

	edges = randomize_edges(edges)

	if len(edges) % 2 != 0:
		logger.info('error?')
		#edges = edges[0:-1]
		# reject
		return generate_network(model)

	edges_new = list()
	for i in range(len(edges)):
		if i > 0 and i % 2 != 0:
			edges_new.append((edges[i-1][0],edges[i][0]))

	# sort edges
	for edge_index in range(len(edges_new)):
		i, j = edges_new[edge_index]
		if j<i:
			edges_new[edge_index] = (j, i)

	# labels
	labels = dict()
	for i in range(model['number_of_nodes']):
		labels[i] = np.random.choice(model['states'], p = model['initial_distribution'])

	generate_nodemap(edges_new)

	graph = (edges_new, labels)

	return graph

#------------------------------------------------------
# Analyze graph
#------------------------------------------------------

def compute_stats(model, g):
	edges, labels = g
	for state in model['states']:
		statecount = len([key for key in labels if labels[key] == state])/float(model['number_of_nodes'])
		model['counter_'+state].append(statecount)

	prev_steps = [0]*(len(model['time'])-1)

	edge_stats = dict()
	for edge in edges:
		n1 = edge[0]
		n2 = edge[1]
		degree1 = len(node_map[n1])
		degree2 = len(node_map[n2])
		label1 = labels[n1]
		label2 = labels[n2]

		edge_id = (label1, label2, degree1, degree2)
		if edge_id not in edge_stats:
			edge_stats[edge_id] = 0
		edge_stats[edge_id]+=1

	model['edge_count'].append(edge_stats)


#------------------------------------------------------
# Compute rates
#------------------------------------------------------

def compute_independent_rates(independent_rules, graph):
	edges, labels = graph
	rates = [-1.0 for r in independent_rules]
	for i in range(len(independent_rules)):
		rule = independent_rules[i]
		candidates = list()
		for node in range(len(labels)):
			if (labels[node]) == (rule[0]):
				candidates.append(node)
		rates[i] = float(rule[2]) * len(candidates)
	return rates

def compute_contact_rates(contact_rules, graph):
	edges, labels = graph
	rates = [-1.0 for r in contact_rules]
	for i in range(len(contact_rules)):
		rule = contact_rules[i]
		candidates = list()
		for edge in edges:
			if ((labels[edge[0]],labels[edge[1]]) == rule[0] or (labels[edge[1]],labels[edge[0]]) == rule[0]):
				candidates.append(edge)
		rates[i] = float(rule[2]) * len(candidates)
	return rates

def compute_rates(model, graph):
	return compute_independent_rates(model['independent_rules'], graph) + compute_contact_rates(model['contact_rules'], graph)

#------------------------------------------------------
# Apply rules
#------------------------------------------------------

def apply_independent_rule(rule, graph):
	edges, labels = graph
	candidates = list()
	for node in range(len(labels)):
		if (labels[node]) == (rule[0]):
			candidates.append(node)

	assert(len(candidates) > 0)

	appply_node = random.choice(candidates)
	labels[appply_node] = rule[1]

def apply_contact_rule(rule, graph):
	edges, labels = graph
	candidates = list()
	for edge in edges:
		if (labels[edge[0]],labels[edge[1]]) == rule[0] or (labels[edge[1]],labels[edge[0]]) == rule[0]:
			candidates.append(edge)

	assert(len(candidates) > 0)

	appply_edge = random.choice(candidates)
	if (labels[edge[0]],labels[edge[1]]) == rule[0] :
		labels[appply_edge[0]] = rule[1][0]
		labels[appply_edge[1]] = rule[1][1]
	else:
		labels[appply_edge[1]] = rule[1][0]
		labels[appply_edge[0]] = rule[1][1]

def apply_rule(rule, graph):
	if str(rule).count(',') <= 2:
		apply_independent_rule(rule, graph)
	else:
		apply_contact_rule(rule, graph)
	pass

#------------------------------------------------------
# Simulation
#------------------------------------------------------

def simulate(model):
	model['max_step'] = 100000
	model['time'] = list()
	model['edge_count'] = list()
	model['rate_sum'] = list()
	for state in model['states']:
		model['counter_'+state] = list()

	logger.info('generate random graph')
	# TODO check equality
	#g = generate_network(model)
	g = create_random_graph(model['number_of_nodes'] , model['degree_distribution'] , model['states'] , model['initial_distribution'])
	logger.info('generation successfull')
	time = 0.0

	while True:
		#print(time)
		#print time
		#print 'current time', time
		model['time'].append(time)
		compute_stats(model, g)
		rates = compute_rates(model, g)
		rate_sum = float(np.sum(rates))
		model['rate_sum'].append(rate_sum)
		if isclose(rate_sum, 0.0):
			print("no reactions possible")
			break
		resid_time = np.random.exponential(1.0 / rate_sum)
		assert(resid_time > 0.0)
		time += resid_time
		normalized_rates = [r/rate_sum for r in rates]
		rules = model['independent_rules'] + model['contact_rules']
		rule_index = np.random.choice(list(range(len(rules))), 1, p=normalized_rates)[0]
		rule = rules[rule_index]
		apply_rule(rule, g)

		if time > model['horizon']:
			print('time is up')
			break
		if 'max_step' in model and model['max_step'] is not None and len(model['time']) >= model['max_step']:
			print('max step is reached')
			break

	return model, g



def write_stats(stats, ci, err_style, model):
	import seaborn as sns
	sns.set_style('white')
	stats = pd.DataFrame(stats)
	stats.to_csv('{}simulation_{}.csv'.format(model['output_dir'], model['name']), sep=',')
	plt.clf()
	sns.tsplot(data=stats, time='Time', unit='unit', condition='State', value='Fraction', estimator=np.nanmean, ci=95)#, ci=ci, err_style=err_style)
	plt.savefig('{}simulation_{}.pdf'.format(model['output_dir'], model['name']))
	#print(stats)

def main(model, runs, nodes, ci, err_style):
	global runtime_storage
	model['number_of_nodes'] = nodes
	stats = {'Fraction': list(), 'unit': list(), 'Time': list(), 'State':list()}
	mean_dict = {s:list() for s in model['states']}
	logger.info('Start Simulations \t'+model['name'])
	start = time.clock()
	for run_i in range(runs):
		model, graph = simulate(model)
		for s in model['states']:
			mean_dict[s].append(list())
		for i, t in enumerate(np.linspace(0.0, model['horizon'], 1001)):
			for s in model['states']:
				timeline = model['time']
				timeline[-1] = model['horizon'] #hack
				smallest_index = np.min([i for i in range(len(timeline)) if timeline[i]>= t])
				value = model['counter_'+s][smallest_index]
				stats['Fraction'].append(value)
				stats['unit'].append(run_i)
				stats['Time'].append(t)
				stats['State'].append(s)
				mean_dict[s][-1].append(value)
		write_stats(stats, ci, err_style, model)

	for s, mean_matrix in mean_dict.items():
		mean_dict[s] = np.mean(mean_matrix, axis=0)
	mean_values = pd.DataFrame(mean_dict)
	mean_values.to_csv('{}simulation_means_{}.csv'.format(model['output_dir'], model['name']), sep=',')
	model['simulation_time'] = time.clock() - start
	with open('{}simulation_times_{}.csv'.format(model['output_dir'], model['name']), 'w') as f:
		f.write(repr(model['simulation_time']))
	logger.info('End Simulations \t'+model['name'])


if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('model',  help="path to modelfile")
	parser.add_argument('--ci', help="confidence interval in %, default is 95", nargs='?')
	parser.add_argument('--nodes', help="number of nodes for generated network, default is 1000", nargs='?')
	parser.add_argument('--runs', help="number of simulation runs, default is 10", nargs='?')
	parser.add_argument('--unit_traces', action='store_true', help="ignore ci and plot individual traces of each run")
	args = parser.parse_args()
	ci = int(args.ci) if args.ci is not None else 95
	runs = int(args.runs) if args.runs is not None else 10
	err_style = 'unit_traces' if args.unit_traces else None
	nodes = int(args.nodes) if args.nodes is not None else 1000

	assert(nodes  > 0)
	assert(ci >0 and ci <100)
	assert(runs > 0)

	model = read_model(args.model)
	model['output_dir']  = os.path.abspath(model['output_dir'])+'/'
	main(model, runs, nodes, ci, err_style)
