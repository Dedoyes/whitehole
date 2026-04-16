import torch
import os.path as path

s = "this is what this write\n yes this is."

file_abs_path = path.abspath (__file__)
src_abs_path = path.dirname (file_abs_path)
base_path = path.dirname (src_abs_path)
data_processed_path = path.join (base_path, "data_processed")
dot_before_path = path.join (data_processed_path, "dot_before")
dot_file_path = path.join (dot_before_path, "0.dot")

x = torch.tensor ([[[1, 1]]])
print (x.shape)
modulus = torch.sqrt (torch.sum (x ** 2))
print (modulus)
