"""Find patchify logic in ZImagePipeline."""
import ast

path = "/usr/local/lib/python3.10/dist-packages/diffusers/pipelines/z_image/pipeline_z_image.py"
with open(path) as f:
    src = f.read()

# Find functions with encoding/processing
tree = ast.parse(src)
print("=== All function names ===")
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef):
        print(f"  {node.lineno}: {node.name}")

# Print the __call__ method
print("\n=== Looking for patch-like operations ===")
for i, line in enumerate(src.split('\n'), 1):
    lower = line.lower()
    if any(w in lower for w in ['patchify', 'unpatchify', 'pack_latent', 'unpack_latent', 'x_embedder', 'view(', '.permute(', '.reshape(']):
        print(f"{i}: {line}")
