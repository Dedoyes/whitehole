SCRIPT_PATH=$(cd "$(dirname "$0")" && pwd)

cd "$SCRIPT_PATH"

clang -Xclang -ast-dump -fsyntax-only ../data/correct_func/func_after/0.c &> ../data/ast.txt
