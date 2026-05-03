"""Find patchify logic in ZImageTransformer2DModel."""
path = "/usr/local/lib/python3.10/dist-packages/diffusers/models/transformers/transformer_z_image.py"
with open(path) as f:
    src = f.read()

print("=== Lines with patch/embed/view/permute/reshape ===")
for i, line in enumerate(src.split('\n'), 1):
    lower = line.lower()
    if any(w in lower for w in ['x_embedder', 'patchify', 'unpatchify', 'pack_latent', '.view(', '.permute(', '.reshape(', 'final_layer', 'all_x_embedder', 'all_final_layer']):
        # Clean up whitespace
        clean = line.strip()
        if clean and not clean.startswith('#'):
            print(f"{i}: {clean}")
