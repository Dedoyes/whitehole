import os.path as path
import os
import numpy as np
import networkx as nx
import pydot

file_abs_path = path.abspath (__file__)
src_abs_path = path.dirname (file_abs_path)
base_abs_path = path.dirname (src_abs_path)
data_abs_path = path.join (base_abs_path, "data")
correct_func_abs_path = path.join (data_abs_path, "correct_func")
dot_abs_path = path.join (correct_func_abs_path, "dot")
func_abs_path = path.join (correct_func_abs_path, "func")
graph_abs_path = path.join (correct_func_abs_path, "graph")

npy_path = path.join (graph_abs_path, "1.npy")

arr = np.load (npy_path)
print (arr)
print (arr.shape)
