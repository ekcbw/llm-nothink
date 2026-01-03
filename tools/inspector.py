import sys, os
sys.path.append(os.path.join(
    os.path.split(__file__)[0],
    "..", "llama.cpp", "gguf-py" # gguf
))
from safetensors import safe_open
from safetensors.torch import save_file
import torch
import gguf

def get_tensor_info(file) -> dict:
    if file.endswith(".gguf"):
        reader = gguf.GGUFReader(file)
        info = {key: field.parts[field.data[0]]
                for key, field in reader.fields.items()}
        info.update({tensor.name:list(tensor.shape)
                     for tensor in reader.tensors})
        return info
    else:
        with safe_open(file, framework="pt") as f:
            return {name:list(f.get_tensor(name).shape) for name in f.keys()}

def get_embedding(file, token_id) -> list:
    if file.endswith(".gguf"):
        reader = gguf.GGUFReader(file)
        tensors = {tensor.name:tensor for tensor in reader.tensors}
        if "output.weight" in tensors:
            row = tensors["output.weight"].data[token_id]
        else:
            row = tensors["token_embd.weight"].data[token_id]
    else:
        with safe_open(file, framework="pt") as f:
            embedding_tensor = f.get_tensor("lm_head.weight")
            row = embedding_tensor[token_id]

    return list(row)

def main():
    if len(sys.argv) == 2:
        file = sys.argv[1]
        print(f"Tensors from {file}:")
        for name, shape in get_tensor_info(file).items():
            print(f"{name}: {shape}")
    elif len(sys.argv) == 3:
        file = sys.argv[1]
        token_id = int(sys.argv[2])
        row = get_embedding(file, token_id)
        format = ', '.join(f"{value:.3g}" for value in row)
        print(f"[{format}]")
    elif len(sys.argv) == 4:
        file = sys.argv[1]
        token_id1 = int(sys.argv[2])
        token_id2 = int(sys.argv[3])
        vec1 = torch.Tensor(get_embedding(file, token_id1))
        vec2 = torch.Tensor(get_embedding(file, token_id2))
        similarity = torch.dot(vec1, vec2).item()
        print(f"{similarity:.6f}")
    else:
        print(f"""Usage: python {sys.argv[0]} <safetensor/gguf file>
           List the tensor names and shapes in the file.
       python {sys.argv[0]} <safetensor/gguf file> <token id>
           View the embedding vector for a token.
       python {sys.argv[0]} <safetensor/gguf file> <token id1> <token id2>
           View the dot product similarity between two tokens.
Examples: python {sys.argv[0]} model.safetensors 151667
          python {sys.argv[0]} model.safetensors 151667 151668""")
        sys.exit(1)

if __name__ == "__main__":
    main()
