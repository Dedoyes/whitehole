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

struct AST_node {
    int id;
    int kind;
    string ast_kind;
    string content;
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

bool type_identify (string s) {
    // true : "12525" [label = <...>]
    // false : "12356" -> "125987"
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
    if (s[pos] == '[')
        return true;
    return false;
}

void extract (string s) {
    if (type_identify (s)) {
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
    } else {
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
        cout << f[u] << " " << f[v] << endl;
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
    string filePath = argv[1];
    cout << "filePath = " << filePath << endl;
    const char* fileName = filePath.c_str ();
    freopen (fileName, "r", stdin);
    cout << "fileName = " << fileName << endl;
    string s;
    while (getline (cin, s)) {
        if (!is_useful (s))
            continue;
        //cout << "s = " << s << endl; 
        extract (s);
    }
    return 0;
}
