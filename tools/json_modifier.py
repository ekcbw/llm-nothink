import sys, json

if len(sys.argv) < 2:
    print(f"""Usage: python {sys.argv[0]} <json file> <operations>
Example: python {sys.argv[0]} config.json "data['tie_word_embeddings']=False"\
""")
    sys.exit(1)

with open(sys.argv[1], encoding="utf-8") as f:
    data = json.load(f)

scope = {"__builtins__": {}, "data": data}
print(f"Modifying {sys.argv[1]}")
for code in sys.argv[2:]:
    try:exec(code, scope)
    except Exception:
        print(f"Failed while executing {code!r}", file=sys.stderr)
        raise

print(f"Saving to {sys.argv[1]}")
with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)