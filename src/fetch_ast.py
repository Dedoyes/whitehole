import sys
sys.path.append ("/usr/lib/python3/dist-packages")

from clang import cindex
import networkx as nx
import os.path as path
import matplotlib.pyplot as plt
import numpy as np
import pydot

dir_abs_path = path.dirname (path.abspath (__file__))
base_abs_path = path.dirname (dir_abs_path)
data_abs_path = path.join (base_abs_path, "data")
correct_func_abs_path = path.join (data_abs_path, "correct_func")
func_after_correct_abs_path = path.join (correct_func_abs_path, "func_after")
func_before_correct_abs_path = path.join (correct_func_abs_path, "func_before")
error_func_abs_path = path.join (data_abs_path, "error_func")
func_after_error_abs_path = path.join (error_func_abs_path, "func_after")
func_before_error_abs_path = path.join (error_func_abs_path, "func_before")
dot_abs_path = path.join (base_abs_path, "check_rodc_critical_attribute.dot")

print ("dotabs_path = ", dot_abs_path)

graphs = pydot.graph_from_dot_file (dot_abs_path)
graph = graphs[0]

nodes = graph.get_nodes ()
nodes_name = []
for node in nodes :
    nodes_name.append (node.get_name ())

print (nodes_name)

size = len (nodes_name)
G = np.zeros ((size, size), dtype=int)
name_to_idx = {}

for i in range (0, size) :
    name = nodes_name[i]
    name_to_idx[name] = i

for edge in graph.get_edges () :
    src = name_to_idx[edge.get_source ()]
    dst = name_to_idx[edge.get_destination ()]
    G[src, dst] = 1

print (G.shape)
print (G)
