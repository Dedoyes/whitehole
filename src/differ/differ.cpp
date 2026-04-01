#include <bits/stdc++.h>
#define LL long long 
#define mod %

using namespace std;

const int MAXN = 2e6;

bool is_num (string s) {
    if (s[0] >= '0' && s[0] <= '9')
        return true;
    return false;
}

int str_to_int (string s) {
    int ret = 0;
    for (int i = 0; i < s.size (); i++)
        ret = ret * 10 + s[i] - '0';
    return ret;
}

struct ASTnode {
    int id, astid;
    string ast_kind, content;
    ASTnode () {
        id = astid = -1;
        ast_kind = content = "";
    }
    ASTnode (int _id, int _astid, string _ast_kind, string _content) {
        id = _id;
        astid = _astid;
        ast_kind = _ast_kind;
        content = _content;
    }
    void print () {

    }
};

struct Diff {
    int type;
    // 0 -> addNode
    // 1 -> delNode
    // 2 -> addEdge
    // 3 -> delEdge
    int u, v;  // Edge start and end
    int id, ast_kind; // Node
    string content;  // add node content
};

struct AST {
    int n;
    ASTnode node[MAXN];
    int fa[MAXN];
    vector <int> G[MAXN];
};

void readFile (AST& tree, ifstream& is) {
    int cnt = 0;
    int id = 0, astid = 0, u = 0, v = 0;
    string ast_kind = "", content = "";
    while (is >> id >> ast_kind) {
        if (!is_num (ast_kind)) {
            is >> astid;
            getline (is, content);
            tree.node[++cnt] = ASTnode (id, astid, ast_kind, content);
        } else {
            u = id;
            v = str_to_int (ast_kind);
            tree.G[u][v] = tree.G[v][u] = 1;
        }
    }
}

int main (int argc, char* argv[]) {
    if (argc < 4) {
        cout << "Error : differ input file lacked" << endl;
        exit (1);
    }
    const char* input_before_path = argv[1];
    const char* input_after_path = argv[2];
    const char* output_path = argv[3];
    ifstream fi_before (input_before_path);
    ifstream fi_after (input_after_path);
    AST tree_before, tree_after;
    readFile (tree_before, fi_before);
    readFile (tree_after, fi_after);
    return 0;
}
