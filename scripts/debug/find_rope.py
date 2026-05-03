"""Find RoPE functions in diffusers."""
import ast, sys
path = "/usr/local/lib/python3.10/dist-packages/diffusers/models/transformers/transformer_z_image.py"
with open(path) as f:
    source = f.read()

tree = ast.parse(source)
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef):
        if 'rotary' in node.name.lower() or 'rope' in node.name.lower():
            print(f"Function: {node.name} at line {node.lineno}")
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Attribute):
            if 'view_as_real' in node.func.attr:
                print(f"view_as_real at line {node.lineno}")
        if isinstance(node.func, ast.Name):
            if node.func.id == 'apply_rotary_emb':
                print(f"apply_rotary_emb call at line {node.lineno}")

# Also just grep the source
print("\n--- Lines with 'rotary' or 'view_as_real' ---")
for i, line in enumerate(source.split('\n'), 1):
    if 'rotary' in line.lower() or 'view_as_real' in line:
        print(f"{i}: {line}")
