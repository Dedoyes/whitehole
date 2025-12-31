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

print ("dot path = ", dot_abs_path)
print ("graph path = ", graph_abs_path)

def get_prefix (s) :
    ret = ""
    for i in range (0, s.__len__ () - 4) :
        ret += s[i]
    return ret

max_graph_size = 0

for filename in os.listdir (dot_abs_path) :
    if filename.endswith (".dot") :
        prefix = get_prefix (filename)
        dot_file_path = path.join (dot_abs_path, filename)
        npyname = prefix + ".npy"
        npy_file_path = path.join (graph_abs_path, npyname)
        graphs = pydot.graph_from_dot_file (dot_file_path)
        graph = graphs[0]
        nodes = graph.get_nodes ()
        nodes_name = []
        for node in nodes :
            nodes_name.append (node.get_name ())
        size = len (nodes_name)
        G = np.zeros ((size, size), dtype=np.float32)
        name_to_idx = {}
        for i in range (0, size) :
            name = nodes_name[i]
            name_to_idx[name] = i
        for edge in graph.get_edges () :
            src = name_to_idx[edge.get_source ()]
            dst = name_to_idx[edge.get_destination ()]
            G[src, dst] = 1.0
        max_graph_size = max (max_graph_size, size)
        np.save (npy_file_path, G)
        print (f"[OK] {npyname} was converted success.")

print ("max_graph_size = ", max_graph_size)
