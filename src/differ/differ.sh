SCRIPT_PATH=$(realpath "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
SRC_DIR=$(dirname "$SCRIPT_DIR")
BASE_DIR=$(dirname "$SRC_DIR")
PROCESSED_DIR="$BASE_DIR/data_processed"
DOT_AFTER_DIR="$PROCESSED_DIR/dot_after"
DOT_BEFORE_DIR="$PROCESSED_DIR/dot_before"
DOT_CORRECT_DIR="$PROCESSED_DIR/dot_correct"
DIFF_GENERATE_DIR="$PROCESSED_DIR/diff"
DIFF_TEST_DIR="$PROCESSED_DIR/diff_test"
SRC_DIR="$BASE_DIR/src"
DIFFER_DIR="$SRC_DIR/differ"
CPP_PATH="$DIFFER_DIR/differ.cpp"
OBJ_PATH="$DIFFER_DIR/differ"
TEST_INPUT_BEFORE_PATH="$DOT_BEFORE_DIR/0.dot"
TEST_INPUT_AFTER_PATH="$DOT_AFTER_DIR/0.dot"
TEST_OUTPUT_PATH="$DIFF_TEST_DIR/diff.dot"

g++ $CPP_PATH -o $OBJ_PATH

$OBJ_PATH $TEST_INPUT_BEFORE_PATH $TEST_INPUT_AFTER_PATH $TEST_OUTPUT_PATH

echo $DOT_AFTER_DIR

num=0
for file in "$DOT_AFTER_DIR/"* 
do 
    #echo $file
    ((num++))
done

echo $num

for ((i=0; i<num; i++))
do
    file_before="$DOT_BEFORE_DIR/$i.dot"
    file_after="$DOT_AFTER_DIR/$i.dot"
    file_out="$DIFF_GENERATE_DIR/$i.dot"
    $OBJ_PATH $file_before $file_after $file_out
    ret=$?
    if [ $ret -ne 0 ]; then
        echo "differ run fail!"
        echo "the number is $i" 
        exit 1
    fi
    progress=$(echo "scale = 4; 100 * $i / $num" | bc)
    #echo $file_before
    #echo $file_after
    #echo $file_out
done
