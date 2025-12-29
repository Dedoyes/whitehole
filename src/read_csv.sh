SCRIPT_PATH=$(cd "$(dirname "$0")" && pwd)

cd "$SCRIPT_PATH"

python3 ./read_csv.py &> ../data/read_csv.out 
