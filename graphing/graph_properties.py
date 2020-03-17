#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar 13 2020

@author: James_Allen
"""

import numpy
import matplotlib.pyplot as plt
import sys

#Returns dictionary of label-numpy array pairs
#Assumes all data are floats
def read_file(file_name):
	file = open(file_name, "r")
	first_line = file.readline()
	labels = first_line.split("|")
	num_labels = len(labels)
	line = "dummy"
	data_table = [[] for label in labels] #Array of arrays corresponding to the data we'll read
	while(line):
		line = file.readline()
		if not line:
			break
		line_parts = line.split("|")
		if(len(line_parts) < num_labels):
			print("Warning: {0} has less elements than {1}".format(line, first_line))
			continue
		for array_index in range(num_labels):
			data_table[array_index].append(float(line_parts[array_index]))
	return_dictionary = {labels[array_index]: numpy.array(data_table[array_index]) for array_index in range(num_labels)}
	file.close()
	return return_dictionary

if len(sys.argv) <= 1:
	print("Need input file to read data from")
	sys.exit(1)
properties_file_name = sys.argv[1]
data_dictionary = read_file(properties_file_name)
color_array = ["#D00000", "#008000", "#0000FF", "#7F00FF", "#FF4000","#202020"]

num_labels = len(data_dictionary)
fig, axes = plt.subplots(num_labels, figsize = (9, 1.5*num_labels), sharex = True)

axis_index = 0
for label in data_dictionary:
	ax = axes[axis_index]
	data = data_dictionary[label]
	iterations = range(len(data))
	col = color_array[axis_index]
	ax.plot(iterations, data, color = col)
	ax.set_ylabel(label, color = col, fontsize = 8)
	axis_index += 1

axes[num_labels-1].set_xlabel("Iteration number")

fig.savefig("../plots/{0}".format(properties_file_name.replace("../out/", "")))


