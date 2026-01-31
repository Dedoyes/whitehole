SCRIPT_PATH=$(realpath "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
program="/label_counter"
source="/label_counter.cpp"
PROGRAM_PATH=$SCRIPT_DIR$program
SOURCE_PATH=$SCRIPT_DIR$source
g++ "$SOURCE_PATH" -o "$PROGRAM_PATH" && "$PROGRAM_PATH"
