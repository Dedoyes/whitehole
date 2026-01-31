#include <bits/stdc++.h>
#define LL long long 
#define mod %

using namespace std;

vector <string> useless_str;

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
        cout << "s = " << s << endl; 
    }
    return 0;
}
