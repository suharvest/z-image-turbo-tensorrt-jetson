path = "/usr/local/lib/python3.10/dist-packages/diffusers/models/transformers/transformer_z_image.py"
with open(path) as f:
    lines = f.readlines()
for i in range(95, 145):
    print(f"{i+1}: {lines[i]}", end="")
