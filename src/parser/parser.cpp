#include <bits/stdc++.h>
#define LL long long 
#define mod %

using namespace std;

const int MAXN = 2e6;

bool is_num (char c) {
    if (c >= '0' && c <= '9')
        return true;
    return false;
}

bool is_letter (char c) {
    if (c >= 'A' && c <= 'Z')
        return true;
    if (c >= 'a' && c <= 'z')
        return true;
    return false;
}

struct AST_node {
    int id;                    // node index
    int kind;                  // node kind
    string ast_kind;           // ast node kind
    string content;            // function content
    AST_node () {
        id = kind = -1;
        ast_kind = content = "";
    }
    AST_node (int _id, int _kind, string _ast_kind, string _content) {
        id = _id;
        kind = _kind;
        ast_kind = _ast_kind;
        content = _content;
    }
};

int cnt = 0;
vector <int> G[MAXN];
vector <string> useless_str;
map <string, int> f;            // ast node id -> simplified id

void init () {
    useless_str.push_back ("digraph");
    useless_str.push_back ("node");
    useless_str.push_back ("}");
}

bool str_start (string pat, string s) {
    if (pat.size () < s.size ())
        return false;
    for (int i = 0; i < s.size (); i++)
        if (pat[i] != s[i])
            return false;
    return true;
}

bool is_useful (string s) {
    for (auto str : useless_str) {
        if (str_start (s, str)) {
            return false;
        }
    }
    return true;
}

int type_identify (string s) {
    // 2 : "68719" [label = <IDENTIFIER  (result) = Min(bitlen, atttypmod)> ]
    // 2 : "23663" [label = <BLOCK <empty>> ]
    // 1 : "12525" [label = <PARAM, 1 int flags> ]
    // 1 : "30064" [label = <<operator>.assignment, 7 need_acl_check = false> ]
    // 0 : "12356" -> "125987"
    int pos = 1;
    while (true) {
        if (s[pos] == '"')
            break;
        pos++;
    }
    pos++;
    while (true) {
        if (s[pos] != ' ')
            break;
        pos++;
    }
    if (s[pos] == '[') {
        int pos = 0;
        for (int i = 0; i < s.size (); i++)
            if (s[i] == '<') {
                pos = i + 1;
                break;
            }
        while (true) {
            if (!is_letter (s[pos])) 
                break;
            pos++;
        }
        if (s[pos] == '<')
            return 1;
        if (s[pos] != ',')
            return 2;
        return 1;
    }
    return 0;
}

void extract (string s) {
    int stence_type = type_identify (s);
    if (stence_type == 1) {
        int pos = 1;
        while (true) {
            if (s[pos] == '"')
                break;
            pos++;
        }
        string num = "", ast_kind = "", content = "", ast_id = "";
        for (int i = 1; i <= pos - 1; i++)
            num += s[i];
        if (!f.count (num)) {
            f[num] = ++cnt;
        }
        int start = 0, end = 0;
        while (true) {
            if (s[pos] == '<')
                break;
            pos++;
        }
        start = ++pos;
        while (true) {
            if (s[pos] == ',')
                break;
            pos++;
        }
        end = pos - 1;
        for (int i = start; i <= end; i++)
            ast_kind += s[i];
        pos++;
        while (true) {
            if (is_num (s[pos])) {
                break;
            }
            pos++;
        }
        start = pos;
        while (true) {
            if (!is_num (s[pos])) {
                break;
            }
            pos++;
        }
        end = pos - 1;
        for (int i = start; i <= end; i++)
            ast_id += s[i];
        start = pos + 1;
        pos = s.size () - 1;
        while (true) {
            if (s[pos] == '>')
                break;
            pos--;
        }
        end = pos - 1;
        for (int i = start; i <= end; i++)
            content += s[i];
        //cout << "num = " << num << endl;
        cout << f[num] << " " << ast_kind << " " << ast_id << " " << content << endl;
    } else if (stence_type == 0) {
        int pos = 1;
        int start = 0, end = 0;
        start = 1;
        while (true) {
            if (s[pos] == '"') 
                break;
            pos++;
        }
        end = pos - 1;
        string u = "", v = "";
        for (int i = start; i <= end; i++)
            u += s[i];
        pos++;
        while (true) {
            if (s[pos] == '"')
                break;
            pos++;
        }
        start = ++pos;
        while (true) {
            if (s[pos] == '"')
                break;
            pos++;
        }
        end = pos - 1;
        for (int i = start; i <= end; i++)
            v += s[i];
        //cout << "u = " << u << ", v = " << v << endl;
        if (!f.count (u) || !f.count (v)) {
            cout << "Error : not found corresponding id" << endl;
            exit (1);
        }
        cout << f[u] << " " << f[v] << endl;
    } else if (stence_type == 2) {
        int pos = 1, start = 0, end = 0;
        string node = "", content = "", type = "";
        while (true) {
            if (s[pos] == '"')
                break;
            pos++;
        }
        start = 1;
        end = pos - 1;
        for (int i = start; i <= end; i++)
            node += s[i];
        while (true) {
            if (s[pos] == '<')
                break;
            pos++;
        }
        start = ++pos;
        while (true) {
            if (!is_letter (s[pos])) 
                break;
            pos++;
        }
        end = pos - 1;
        for (int i = start; i <= end; i++)
            type += s[i];
        start = ++pos;
        pos = s.size () - 1;
        while (true) {
            if (s[pos] == '>')
                break;
            pos--;
        }
        end = pos - 1;
        for (int i = start; i <= end; i++)
            content += s[i];
        if (!f.count (node))
            f[node] = ++cnt;
        cout << f[node] << " " << type << " " << 0 << " " << content << endl;
    } else {
        cout << "Error : unknown stence!" << endl;
    }
    return;
}

int main (int argc, char* argv[]) {
    init ();
    int lineNum = 0;
    cout << "argc = " << argc << endl;
    for (int i = 1; i < argc; i++) {
        cout << "argv[" << i << "] = " << argv[i] << endl;
    }
    if (argc < 3) {
        cout << "Error : lack argument" << endl;
        exit (1);
    }
    string filePath = argv[1];
    string outPath = argv[2];
    cout << "filePath = " << filePath << endl;
    const char* fileName = filePath.c_str ();
    const char* outName = outPath.c_str ();
    freopen (fileName, "r", stdin);
    freopen (outName, "w", stdout);
    //cout << "fileName = " << fileName << endl;
    string s;
    while (getline (cin, s)) {
        if (!is_useful (s))
            continue;
        //cout << "s = " << s << endl; 
        extract (s);
    }
    return 0;
}
