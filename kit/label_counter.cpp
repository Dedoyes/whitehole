#include <bits/stdc++.h>
#define LL long long 
#define mod %

using namespace std;

int getPatternSum (string text, string pattern) {
    int tot = 0;
    int len = pattern.size ();
    //cout << "text.size = " << text.size () << endl;
    //cout << "pattern.size = " << pattern.size () << endl;
    //cout << "max i = " << (int)text.size () - len << endl;
    for (int i = 0; i <= (int)text.size () - len; i++) {
        //cout << "i = " << i << endl;
        string subt = "", subp = "";
        for (int j = i; j <= i + len - 1; j++)
            subt += text[j];
        if (subt == pattern)
            tot++;
    }
    return tot;
}

int main () {
    string text;
    int tot = 0;
    while (true) {
        getline (cin, text);
        string pattern = "label";
        tot += getPatternSum(text, pattern);
        cout << "tot = " << tot << endl;
    }
    return 0;
}
