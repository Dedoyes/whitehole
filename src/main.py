import ast

code = """
def add (a, b) : 
    return a + b
"""

tree = ast.parse (code)

print (ast.dump (tree, indent=2))
