import sys, os, multiprocessing
import torch
from safetensors import safe_open
from safetensors.torch import save_file

def erase_token(file, operations, inplace=False):
    output_file = f"{'.'.join(file.split('.')[:-1])}_modified.safetensors"
    if inplace:
        # 避免 Error while serializing: I/O error
        p = multiprocessing.Process(target=erase_token,
                                    args=(file, operations, False))
        p.start(); p.join()
        print(f"Replacing original {file}")
        os.remove(file); os.rename(output_file, file)
        return

    with safe_open(file, framework="pt") as f:
        embedding = f.get_tensor("model.embed_tokens.weight")
        lm_head = f.get_tensor("lm_head.weight")
        scope = {"__builtins__": {}, "lm_head": lm_head,
                 "emb": embedding,
                 "zero": torch.zeros(lm_head.shape[1])}
        print("Modifying tensors")
        for code in operations:
            try:exec(code, scope)
            except Exception:
                print(f"Failed while executing {code!r}", file=sys.stderr)
                raise

        new_weights = {"lm_head.weight": lm_head,
                       "model.embed_tokens.weight": embedding}
        for key in f.keys():
            if key in new_weights:continue
            new_weights[key] = f.get_tensor(key)

    print(f"Saving to {file if inplace else output_file}")
    save_file(new_weights, output_file)

def main():
    if len(sys.argv) < 3:
        print(f"""Usage: python {sys.argv[0]} <safetensor file> <operations>
       Set embedding vector of specified tokens to zero.
Examples: python {sys.argv[0]} model.safetensors \
"lm_head[151668]=lm_head[151667]" lm_head[151667]=zero""")
        sys.exit(1)
    if "--inplace" in sys.argv[1:]:
        inplace = True
        sys.argv.remove("--inplace")
    else:
        inplace = False
    erase_token(sys.argv[1], sys.argv[2:], inplace)

if __name__ == "__main__":main()
