SCRIPT_PATH=$(realpath "$0")
#echo "abspath = " $SCRIPT_PATH
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
#echo "dirpath = " $SCRIPT_DIR

export PATH=$PATH:/home/wbt/newDisk/software/joern/joern-cli

joern --script ./extract.scala
