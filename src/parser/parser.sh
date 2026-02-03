SCRIPT_PATH=$(realpath "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
SRC_DIR=$(dirname "$SCRIPT_DIR")
BASE_DIR=$(dirname "$SRC_DIR")
DATA_DIR="$BASE_DIR/data"
PROCESSED_DIR="$BASE_DIR/data_processed"
OUTPUT_CORRECT_DIR="$PROCESSED_DIR/dot_correct"
OUTPUT_BEFORE_DIR="$PROCESSED_DIR/dot_before"
OUTPUT_AFTER_DIR="$PROCESSED_DIR/dot_after"
CORRECT_DIR="$DATA_DIR/correct_func"
ERROR_DIR="$DATA_DIR/error_func"
AST_DIR="$CORRECT_DIR/ast"
AST_AFTER_DIR="$ERROR_DIR/ast_after"
AST_BEFORE_DIR="$ERROR_DIR/ast_before"
CPP_SOURCE_PATH="$SCRIPT_DIR/parser.cpp"
OBJ_PATH="$SCRIPT_DIR/parser"
TXT_FILE_PATH="$SCRIPT_DIR/test.txt"

echo $AST_AFTER_DIR
echo $CPP_SOURCE_PATH
echo $OBJ_PATH

g++ $CPP_SOURCE_PATH -o $OBJ_PATH

num=0
for file in "$AST_BEFORE_DIR/"*
do
    ((num++))
done

i=0
for file in "$AST_BEFORE_DIR/"*
do 
    echo "the $i th : $file"
    OUTPUT_FILE_PATH="$OUTPUT_BEFORE_DIR/$i.dot"
    echo "$OUTPUT_FILE_PATH"
    $OBJ_PATH $file $OUTPUT_FILE_PATH
    ret=$?
    if [ $ret -ne 0 ]; then
        echo "parser run file!"
        echo "Error file : $file"
        exit 1
    fi
    ((i++))
    progress=$(echo "scale = 4; 100 * $i / $num" | bc)
    clear
    echo "$progress%"
done

exit 0

