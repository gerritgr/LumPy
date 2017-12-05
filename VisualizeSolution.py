"""
Visualization of 2-State solutions.
Input is the .csv file containing all trajectories.
Output is a .gif animation (and single frame files).
You might want to set "eval_points = 100", to avoid an explosion of the filesize.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, os, glob, subprocess

#------------------------------------------------------
# Config and Set-up
#------------------------------------------------------

CSV_SOLUTION_FILE = './output/SIS50/SIS50baseline.csv'
# {} for index with padding zero to ensure lexiographical order
# you can also change the output format here
OUTPATH = 'visualization/output_SI50_{0:07d}.png'
GIFPATH = 'visualization/output_SI50.gif'

WRITE_FRAME_WISE_PICKLE = False
WRITE_FRAME_WISE_CSV = False

# Make sure output directory exists
if not os.path.exists('visualization'):
	os.makedirs('visualization')

created_images = list()

#------------------------------------------------------
# Code to Generate Single Frames
#------------------------------------------------------

def plotx(x_list, y_list, color_list, filename='output.pdf'):
	print(x_list)
	print(y_list)
	print(color_list)

def plot_scatter(x,y, outpath, color = 'r', x_label = 'Infected Neighbours', y_label = 'Susceptible Neighbours'):
	global created_images
	import numpy as np
	import matplotlib
	import pickle
	import matplotlib as mpl
	import matplotlib.pyplot as plt
	import seaborn as sns
	sns.set_style('white')
	mpl.rc('xtick', labelsize=16)
	mpl.rc('ytick', labelsize=16)
	alpha = 0.6
	area = 150.0
	fig, ax1 = plt.subplots()
	ax1.set_xlabel(x_label,fontsize=16)
	ax1.set_xlim([0, max(x)*1.1])
	s1 = ax1.scatter(x, y, s=area, c=color, alpha=alpha, marker = '.', cmap='seismic', vmin=0.0, vmax=0.4, linewidths = 0.5, edgecolors='gray')#, norm=matplotlib.colors.LogNorm())#, norm=matplotlib.colors.LogNorm())
	s2 = ax1.scatter(x, y, s=0.0, c=color, alpha=1.0, marker = ',', cmap='seismic', vmin=0.0, vmax=0.4, linewidths=0.0)#, linewidths=0, norm=matplotlib.colors.LogNorm())
	#s2 = ax1.scatter([1], [1], s=0.0, c=color, alpha=1.0, marker = ',', cmap='YlGnBu', linewidths=0, norm=matplotlib.colors.LogNorm())
	#ax1.legend(fontsize='small')
	ax1.set_ylabel(y_label,fontsize=16)
	ax1.set_ylim([0, max(y)*1.1])
	cb = plt.colorbar(s2)
	cb.solids.set_edgecolor("face")
	cb.outline.set_linewidth(0)
	ax1.set_aspect('equal')
	plt.title('Fraction of Infected Nodes', fontsize=18)
	plt.savefig(outpath, bbox_inches='tight')
	if WRITE_FRAME_WISE_PICKLE:
		pickle.dump(ax1, open(outpath[:-4]+'.pickle', 'wb'))
	if WRITE_FRAME_WISE_PICKLE:
		df = pd.DataFrame({x_label: x, y_label : y})
		df.to_csv(outpath[:-4]+'.csv', header='sep=,')
	created_images.append(outpath)
	plt.close()


data = pd.read_csv(CSV_SOLUTION_FILE, sep=',', skiprows=1)
data = data.to_dict(orient='records')
for i in range(len(data)):
	if not i % 10 == 0:
		continue
	ode = data[i]
	x_list = list()
	y_list = list()
	color_list = list()
	for state_m, value in ode.items():
		state, mean_m = state_m.split('_')
		if state == 'S':
			continue
		mean_m = mean_m.replace(';',',')
		mean_m = eval(mean_m)
		x_list.append(mean_m[0])
		y_list.append(mean_m[1])
		if not value > 0:
			value = 0.0
		color_list.append((value))
	plot_scatter(x_list, y_list, OUTPATH.format(i) ,color_list)


#------------------------------------------------------
# Create .gif Animation from Single Frames
#------------------------------------------------------

import imageio
images = []
for filename in created_images:
	images.append(imageio.imread(filename))
imageio.mimsave(GIFPATH, images)
