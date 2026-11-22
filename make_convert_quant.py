import sys
from pathlib import Path

src_path = Path(r"comfy\custom_nodes\ComfyUI-GGUF\tools\convert.py")
dst_path = Path(r"comfy\custom_nodes\ComfyUI-GGUF\tools\convert_quant.py")

if not src_path.is_file():
    print(f"[Error] Source file not found: {src_path}")
    sys.exit(1)

lines = src_path.read_text(encoding="utf-8").splitlines()
new_lines = []

# 1. Top-level CLI interceptor and default quant setting
header = """# --- FOOOCUSPLUS GGUF QUANTIZATION EXTENSION ---
import sys
TARGET_QUANT = "Q8_0"
if "-q" in sys.argv:
    idx = sys.argv.index("-q")
    TARGET_QUANT = sys.argv[idx + 1]
    sys.argv.pop(idx)
    sys.argv.pop(idx)
elif "--quant" in sys.argv:
    idx = sys.argv.index("--quant")
    TARGET_QUANT = sys.argv[idx + 1]
    sys.argv.pop(idx)
    sys.argv.pop(idx)
# -----------------------------------------------
"""
new_lines.extend(header.splitlines())

# 2. Mixed-precision replacement block
m_recipe_code = [
    "# Resolve target quantization type (supports _M mixed-precision recipes)",
    "t_quant_upper = TARGET_QUANT.upper()",
    "if t_quant_upper in ('Q4_K_M', 'Q5_K_M'):",
    "    base_quant = 'Q4_K' if 'Q4_K' in t_quant_upper else 'Q5_K'",
    "    # Elevate cross-attention value and output projections to Q6_K",
    "    m_elevate = ('attn2.to_v', 'attn2.to_out.0', 'attn1.to_out.0', 'proj_out', 'out_layers')",
    "    if any(k in key for k in m_elevate):",
    "        data_qtype = gguf.GGMLQuantizationType.Q6_K",
    "    else:",
    "        data_qtype = getattr(gguf.GGMLQuantizationType, base_quant)",
    "else:",
    "    clean_quant = t_quant_upper.replace('_S', '')",
    "    data_qtype = getattr(gguf.GGMLQuantizationType, clean_quant, gguf.GGMLQuantizationType.Q8_0)"
]

in_dtype_block = False

for line in lines:
    if "if old_dtype == torch.bfloat16:" in line:
        # Match exact surrounding indentation
        indent = line[:len(line) - len(line.lstrip())]
        for r_line in m_recipe_code:
            new_lines.append(f"{indent}{r_line}")
        in_dtype_block = True
        continue

    if in_dtype_block:
        if "MAX_TENSOR_DIMS" in line or "# The max no. of dimensions" in line:
            in_dtype_block = False
            new_lines.append(line)
        continue

    new_lines.append(line)

dst_path.write_text("\n".join(new_lines), encoding="utf-8")
print(f"[Success] Generated: {dst_path}")