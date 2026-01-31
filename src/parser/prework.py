import re
import os
import html
import os.path as path
import platform

def clear_screen () :
    if platform.system () == "Windows" :
        os.system ("cls")
    else :
        os.system ("clear")

def get_file_tot (dir_path) :
    count = 0
    for file in os.listdir (dir_path) :
        count += 1
    return count

def strip_html_like (text) :
    if not text :
        return text
    text = text.replace ('<BR/>', ' ')
    text = html.unescape (text)
    text = re.sub (r'^<|>$', '', text)
    return text.strip ()

def main () :
    file_path = path.abspath (__file__)
    parser_dir = path.dirname (file_path)
    src_dir = path.dirname (parser_dir)
    base_dir = path.dirname (src_dir)
    data_dir = path.join (base_dir, "data")
    correct_dir = path.join (data_dir, "correct_func")
    dot_dir = path.join (correct_dir, "dot")
    ast_dir = path.join (correct_dir, "ast")
    error_dir = path.join (data_dir, "error_func")
    dot_after_dir = path.join (error_dir, "dot_after")
    dot_before_dir = path.join (error_dir, "dot_before")
    ast_after_dir = path.join (error_dir, "ast_after")
    ast_before_dir = path.join (error_dir, "ast_before")
    text_path = path.join (parser_dir, "test.txt")
    output_path = path.join (parser_dir, "text.out")

    #print ("dot_dir=", dot_dir)
    #print ("ast_dir=", ast_dir)
    #return
    if False :
        count = 0
        dot_file_sum = get_file_tot (dot_dir)
        for filename in os.listdir (dot_dir) :
            count += 1
            dot_input_path = path.join (dot_dir, filename)
            dot_output_path = path.join (ast_dir, filename)
            with open (dot_input_path, "r", encoding="utf-8") as fin, \
                open (dot_output_path, "w", encoding="utf-8") as fout :
                for line in fin :
                    line = line.rstrip ("\n")
                    convert_line = strip_html_like (line) + '\n'
                    fout.write (convert_line)
            percent = 1.0 * count / dot_file_sum
            clear_screen ()
            print (round (100.0 * percent, 2), "%")
    
    if True :
        count = 0
        dot_file_sum = get_file_tot (dot_after_dir)
        for filename in os.listdir (dot_after_dir) :
            count += 1
            dot_input_path = path.join (dot_after_dir, filename)
            dot_output_path = path.join (ast_after_dir, filename)
            with open (dot_input_path, "r", encoding="utf-8") as fin, \
                open (dot_output_path, "w", encoding="utf-8") as fout :
                for line in fin :
                    line = line.rstrip ("\n")
                    convert_line = strip_html_like (line) + '\n'
                    fout.write (convert_line)
            percent = 1.0 * count / dot_file_sum
            clear_screen ()
            print (round (100.0 * percent, 2), "%")
    
    if True :
        count = 0
        dot_file_sum = get_file_tot (dot_before_dir)
        for filename in os.listdir (dot_before_dir) :
            count += 1
            dot_input_path = path.join (dot_before_dir, filename)
            dot_output_path = path.join (ast_before_dir, filename)
            with open (dot_input_path, "r", encoding="utf-8") as fin, \
                open (dot_output_path, "w", encoding="utf-8") as fout :
                for line in fin :
                    line = line.rstrip ("\n")
                    convert_line = strip_html_like (line) + '\n'
                    fout.write (convert_line)
            percent = 1.0 * count / dot_file_sum
            clear_screen ()
            print (round (100.0 * percent, 2), "%")

    return 0

if __name__ == "__main__" :
    main ()
