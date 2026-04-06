#include <bits/stdc++.h>
#include <fstream>
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
    void print (ofstream& stream) {
        stream << id << " " << astid << " " << ast_kind << " " << content << endl;
    }
    bool operator == (const ASTnode& node) const {
        return (this->id == node.id) && (this->astid == node.astid) &&
            (this->ast_kind == node.ast_kind) && (this->content == node.content);
    }
    bool operator != (const ASTnode& node) const {
        return !(*this == node);
    }
};

struct Diff {
    int type;
    // 0 -> addNode
    // 1 -> delNode
    // 2 -> addEdge
    // 3 -> delEdge
    int u, v;  // Edge start and end
    ASTnode node; // add or delete AST node
    void print (ofstream& stream) {
        if (this->type == 0) {
            stream << "add ";
            this->node.print (stream);
        } else if (this->type == 1) {
            stream << "delete ";
            this->node.print (stream);
        } else if (this->type == 2) {
            stream << "addEdge " << u << " " << v << endl;
        } else if (this->type == 3) {
            stream << "deleteEdge : " << u << " " << v << endl;
        } else {
            cout << "Diff type Error !";
            exit (1);
        }
    }
};

struct AST {
    int n;
    ASTnode node[MAXN];
    set <int> G[MAXN];     // linked forward star
    void print (ofstream& stream) {
        cout << "n = " << n << endl;
        for (int i = 1; i <= n; i++) {
            node[i].print (stream);
        }
        for (int i = 1; i <= n; i++) {
            for (auto x : G[i]) {
                cout << i << "->" << x << endl;
            }
        }
    }
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
            tree.G[u].insert (v);
        }
    }
    tree.n = cnt;
    //cout << "tree.n = " << tree.n << endl;
}

vector <Diff> getDiff (AST& tree_before, AST& tree_after) {
    vector <Diff> ret;
    map <int, bool> isdel;
    for (int i = 1; i <= tree_after.n; i++) {
        if (i <= tree_before.n) {
            if (tree_before.node[i] != tree_after.node[i]) {
                isdel[i] = true;
                Diff diff;  
                diff.type = 1;
                diff.node = tree_before.node[i];
                diff.u = diff.v = 0;
                ret.push_back (diff);
                diff.type = 0;
                diff.node = tree_after.node[i];
                diff.u = diff.v = 0;
                ret.push_back (diff);
            }
        } else {
            Diff diff;
            diff.type = 0;
            diff.node = tree_after.node[i];
            diff.u = diff.v = 0;
            ret.push_back (diff);
        }
    }
    for (int i = tree_after.n + 1; i <= tree_before.n; i++) {
        isdel[i] = true;
        Diff diff;
        diff.type = 1;
        diff.node = tree_before.node[i];
        diff.u = diff.v = 0;
        ret.push_back (diff);
    }
    for (int i = 1; i <= tree_before.n; i++) {
        if (isdel[i]) {
            for (auto x : tree_before.G[i]) {
                Diff diff;
                diff.type = 3;
                diff.u = i;
                diff.v = x;
                ret.push_back (diff);
            }
            tree_before.G[i].clear ();
        }
    }
    for (int u = 1; u <= tree_after.n; u++) {
        for (auto v : tree_after.G[u]) {
            auto it = tree_before.G[u].find (v);
            if (it != tree_before.G[u].end ()) {
                continue;
            } else {
                Diff diff;
                diff.type = 2;
                diff.u = u;
                diff.v = v;
                ret.push_back (diff);
            }
        }
    }
    return ret;
}

AST tree_before, tree_after;

int main (int argc, char* argv[]) {
    cout << "program start" << endl;
    if (argc < 4) {
        cout << "Error : differ input file lacked" << endl;
        exit (1);
    }
    const char* input_before_path = argv[1];
    const char* input_after_path = argv[2];
    const char* output_path = argv[3];
    cout << "input_before_path = " << input_before_path << endl;
    cout << "input_after_path = " << input_after_path << endl;
    cout << "output_path = " << output_path << endl;
    ifstream fi_before (input_before_path);
    ifstream fi_after (input_after_path);
    ofstream fi_out (output_path);
    readFile (tree_before, fi_before);
    readFile (tree_after, fi_after);
    //tree_before.print ();
    //tree_after.print ();
    vector <Diff> diff_vec = getDiff (tree_before, tree_after);
    for (auto diff : diff_vec) {
        diff.print (fi_out);
    }
    return 0;
}
