import os.path as path

s = "this is what this write\n yes this is."

file_abs_path = path.abspath (__file__)
src_abs_path = path.dirname (file_abs_path)
write_abs_path = path.join (src_abs_path, "1.c")

with open (write_abs_path, "w") as f :
    f.write (s)
