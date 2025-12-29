import pandas
import os.path as path

dir_abs_path = path.dirname (path.abspath (__file__))
base_abs_path = path.dirname (dir_abs_path)
data_abs_path = path.join (base_abs_path, "data")
csv_abs_path = path.join (data_abs_path, "MSR_data_cleaned.csv")
output_abs_path = path.join (data_abs_path, "output.csv")
error_func_abs_path = path.join (data_abs_path, "error_func")
correct_func_abs_path = path.join (data_abs_path, "correct_func")
func_after_abs_path = path.join (error_func_abs_path, "func_after")
func_before_abs_path = path.join (error_func_abs_path, "func_before")
norm_func_after_abs_path = path.join (correct_func_abs_path, "func_after")
norm_func_before_abs_path = path.join (correct_func_abs_path, "func_before")

pandas.set_option ('display.max_colwidth', None)
pandas.set_option ('display.width', None)

print (csv_abs_path)

file = pandas.read_csv (csv_abs_path)

len = file.shape[0]

#print (len)

error_sum = 0
correct_sum = 0
list = []

for i in range (0, len) :
    if file.iloc[i]["lang"] == "C" :
        func_before = file.iloc[i]["func_before"]
        func_after = file.iloc[i]["func_after"]
        vul = file.iloc[i]["vul"]
        #row_select = file.iloc[i]["func_before", "func_after", "vul"]
        list.append ([func_before, func_after, vul])
        if vul == 0 :
            filename = str (correct_sum) + ".c"
            norm_file_before_abs_path = path.join (norm_func_before_abs_path, filename)
            norm_file_after_abs_path = path.join (norm_func_after_abs_path, filename)
            with open (norm_file_before_abs_path, "w") as f :
                f.write (func_before)
            with open (norm_file_after_abs_path, "w") as f :
                f.write (func_after)
            correct_sum += 1
        else :
            filename = str (error_sum) + ".c"
            file_before_abs_path = path.join (func_before_abs_path, filename)
            file_after_abs_path = path.join (func_after_abs_path, filename)
            with open (file_before_abs_path, "w") as f :
                f.write (func_before)
            with open (file_after_abs_path, "w") as f :
                f.write (func_after)
            error_sum += 1
#print (list)

df = pandas.DataFrame (
    list,
    columns=["func_before", "func_after", "vul"]
)

df.to_csv (output_abs_path, index=True)
print ("num = ", len)
print ("error = ", error_sum)
print ("correct = ", correct_sum)
