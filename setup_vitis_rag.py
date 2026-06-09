"""
Seed the Vitis AI compatibility Chroma DB.
Full per-architecture constraint data:
  kernel size limits, stride constraints, channel width limits,
  dilation support, padding rules, fused patterns, and board mappings.

Architecture family → short ID used in metadata:
  DPUCZDX8G (Zynq UltraScale+): B512 B800 B1024 B1152 B1600 B2304 B3136 B4096
  DPUCVDX8G  — Versal AI Core   (VCK190)
  DPUCVDX8H  — Versal AI Edge   (VEK280 HBM)
  DPUCAHX8H  — Alveo U50/U200/U250/U280
  DPUCADF8H  — Alveo U55C / V70 (data-centre)

Board → arch mapping:
  KV260, ZCU102, ZCU106 → B4096
  ZCU104               → B2304
  Ultra96-v2           → B1600
  Kria K26 SOM         → B4096
  VCK190               → DPUCVDX8G
  VEK280               → DPUCVDX8H
  Alveo U50/U200/U250  → DPUCAHX8H
  Alveo U55C / V70     → DPUCADF8H
"""
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import shutil, os

# ---------------------------------------------------------------------------
# Helper — build one Document per arch × constraint block
# ---------------------------------------------------------------------------
def d(content: str, op: str, arch: str, supported: bool, severity: str = "ok") -> Document:
    return Document(
        page_content=content,
        metadata={"op": op, "arch": arch, "supported": supported, "severity": severity},
    )

VITIS_DOCS = [

    # ═══════════════════════════════════════════════════════════════════════
    # Conv2d / DepthwiseConv
    # ═══════════════════════════════════════════════════════════════════════
    d("Conv on B512: max in/out channels 128. Kernel 1×1–4×4. Stride 1 or 2. No dilation. No depthwise. Groups > 1 NOT supported. Padding: symmetric only.",
      "Conv", "B512", True, "warning"),
    d("Conv on B800: max channels 192. Kernel 1×1–6×6. Stride 1 or 2. Dilation NOT supported. Depthwise (groups=channels) IS supported from B800 onward.",
      "Conv", "B800", True, "ok"),
    d("Conv on B1024: max channels 256. Kernel 1×1–8×8. Stride 1, 2, or 4. No dilation. Depthwise supported. Asymmetric padding NOT supported.",
      "Conv", "B1024", True, "ok"),
    d("Conv on B1152: max channels 288. Kernel 1×1–8×8. Stride 1, 2, or 4. Depthwise supported.",
      "Conv", "B1152", True, "ok"),
    d("Conv on B1600: max channels 384. Kernel 1×1–8×8. Stride 1, 2, or 4. Depthwise supported. Dilated conv NOT supported.",
      "Conv", "B1600", True, "ok"),
    d("Conv on B2304: max channels 512. Kernel 1×1–8×8. Stride 1, 2, or 4. Depthwise supported. No dilation.",
      "Conv", "B2304", True, "ok"),
    d("Conv on B3136: max channels 768. Kernel 1×1–12×12. Stride 1, 2, or 4. Depthwise supported.",
      "Conv", "B3136", True, "ok"),
    d("Conv on B4096: max channels 1024. Kernel 1×1–16×16. Stride 1, 2, 4, or 8. Depthwise supported. Dilated conv (atrous) NOT supported — replace with regular conv or ASPP alternative.",
      "Conv", "B4096", True, "ok"),
    d("Conv on DPUCVDX8G (VCK190): no channel limit. Kernel up to 16×16. Stride 1–8. Dilated conv supported with dilation ≤ 4. Depthwise supported.",
      "Conv", "DPUCVDX8G", True, "ok"),
    d("Conv on DPUCVDX8H (VEK280): no channel limit. Same as DPUCVDX8G but higher throughput via HBM bandwidth.",
      "Conv", "DPUCVDX8H", True, "ok"),
    d("Conv on DPUCAHX8H (Alveo U50/U200/U280): no practical channel limit. All kernel sizes and strides. Dilated conv supported.",
      "Conv", "DPUCAHX8H", True, "ok"),
    d("Conv on DPUCADF8H (Alveo U55C/V70): no channel limit. Optimised for large batch throughput. All kernel sizes. Dilated conv supported.",
      "Conv", "DPUCADF8H", True, "ok"),

    # ═══════════════════════════════════════════════════════════════════════
    # MaxPool
    # ═══════════════════════════════════════════════════════════════════════
    d("MaxPool on B512: max kernel 2×2 only. Stride 1 or 2. Any larger kernel falls to CPU. Replace SPPF 5×5 with stride-2 Conv.",
      "MaxPool", "B512", False, "critical"),
    d("MaxPool on B800: max kernel 3×3. Stride 1 or 2. SPPF 5×5 MaxPool NOT supported — cascade three 3×3 MaxPools.",
      "MaxPool", "B800", True, "warning"),
    d("MaxPool on B1024: max kernel 4×4. Stride 1 or 2. SPPF 5×5 NOT supported. Use cascaded 3×3 or replace SPPF block.",
      "MaxPool", "B1024", False, "critical"),
    d("MaxPool on B1152: max kernel 4×4. SPPF 5×5 NOT supported. Stride 1 or 2 only.",
      "MaxPool", "B1152", False, "critical"),
    d("MaxPool on B1600: max kernel 6×6. SPPF 5×5 MaxPool IS supported. Stride 1 or 2.",
      "MaxPool", "B1600", True, "ok"),
    d("MaxPool on B2304: max kernel 6×6. SPPF 5×5 supported. 7×7 falls to CPU. Stride 1 or 2.",
      "MaxPool", "B2304", True, "warning"),
    d("MaxPool on B3136: max kernel 8×8. SPPF fully supported. Stride 1 or 2.",
      "MaxPool", "B3136", True, "ok"),
    d("MaxPool on B4096: max kernel 8×8. SPPF with three 5×5 MaxPools fully supported. Stride 1 or 2. Dilation NOT supported.",
      "MaxPool", "B4096", True, "ok"),
    d("MaxPool on DPUCVDX8G: max kernel 16×16. All common SPPF configurations supported. Stride 1–4.",
      "MaxPool", "DPUCVDX8G", True, "ok"),
    d("MaxPool on DPUCVDX8H: max kernel 16×16. All SPPF configurations fully supported.",
      "MaxPool", "DPUCVDX8H", True, "ok"),
    d("MaxPool on DPUCAHX8H and DPUCADF8H: max kernel 16×16. All configurations supported.",
      "MaxPool", "DPUCAHX8H", True, "ok"),

    # ═══════════════════════════════════════════════════════════════════════
    # AveragePool / GlobalAveragePool / AdaptiveAvgPool
    # ═══════════════════════════════════════════════════════════════════════
    d("AveragePool on B512–B1024: fixed kernel only. Max kernel 4×4. Stride 1 or 2. GlobalAveragePool IS supported. AdaptiveAvgPool NOT supported — replace with fixed AveragePool.",
      "AveragePool", "B1024", True, "warning"),
    d("AveragePool on B2304–B4096: fixed kernel up to 8×8. GlobalAveragePool supported. Stride 1 or 2.",
      "AveragePool", "B4096", True, "ok"),
    d("GlobalAveragePool is supported on all DPUCZDX8G variants (B512–B4096) and DPUCVDX8G/H.",
      "GlobalAveragePool", "ALL", True, "ok"),
    d("AdaptiveAvgPool is NOT supported on any DPU architecture. Replace with fixed-size AveragePool(kernel_size=H//out_H, stride=H//out_H).",
      "AdaptiveAvgPool", "ALL", False, "critical"),
    d("AveragePool on DPUCVDX8G/H: fixed or global, all kernel sizes up to 16×16 supported.",
      "AveragePool", "DPUCVDX8G", True, "ok"),

    # ═══════════════════════════════════════════════════════════════════════
    # Sigmoid / SiLU (Sigmoid×Mul pattern in ONNX)
    # ═══════════════════════════════════════════════════════════════════════
    d("Sigmoid on B512/B800/B1024/B1152: NOT supported mid-graph. Forces DPU subgraph boundary. Output-only Sigmoid (confidence scores) is tolerated but will run on CPU. SiLU (Sigmoid×x) must be replaced with ReLU or HardSwish.",
      "Sigmoid", "B1024", False, "critical"),
    d("Sigmoid on B1600/B2304/B3136/B4096: NOT DPU-native mid-graph. SiLU (Sigmoid followed by element-wise Mul) appears in every YOLOv8 C2f block and causes repeated DPU↔CPU transitions. Replace with HardSwish (HardSigmoid×x) or LeakyReLU(0.1).",
      "Sigmoid", "B4096", False, "critical"),
    d("Sigmoid on DPUCVDX8G/H: limited mid-graph support. End-of-graph Sigmoid is acceptable. For YOLOv8 backbone SiLU, replace with HardSwish to keep full backbone on DPU.",
      "Sigmoid", "DPUCVDX8G", False, "warning"),
    d("Sigmoid on DPUCAHX8H/DPUCADF8H (Alveo): mid-graph Sigmoid supported in limited fused patterns. SiLU still recommended to replace with HardSwish for best throughput.",
      "Sigmoid", "DPUCAHX8H", True, "warning"),

    # ═══════════════════════════════════════════════════════════════════════
    # Softmax
    # ═══════════════════════════════════════════════════════════════════════
    d("Softmax is NOT supported on any DPUCZDX8G DPU (B512–B4096). Always runs on ARM CPU. Move Softmax to post-processing. For YOLOv8 DFL head: compute Softmax + weighted sum on CPU after DPU inference.",
      "Softmax", "B4096", False, "critical"),
    d("Softmax is NOT supported on DPUCVDX8G or DPUCVDX8H DPU. Must run on ARM CPU.",
      "Softmax", "DPUCVDX8G", False, "critical"),
    d("Softmax is NOT supported on DPUCAHX8H or DPUCADF8H DPU. Runs on host CPU.",
      "Softmax", "DPUCAHX8H", False, "critical"),

    # ═══════════════════════════════════════════════════════════════════════
    # Resize / Upsample
    # ═══════════════════════════════════════════════════════════════════════
    d("Resize on B512/B800: nearest-neighbor mode only. Integer scale factor 2× only. Dynamic output size NOT supported. Bilinear NOT supported.",
      "Resize", "B800", True, "warning"),
    d("Resize on B1024/B1152/B1600: nearest-neighbor with integer scale factors (2× or 4×). Output size must be static at export. Bilinear NOT supported. Dynamic scales cause CPU fallback.",
      "Resize", "B1600", True, "warning"),
    d("Resize on B2304/B3136/B4096: nearest-neighbor with fixed integer scales (2× or 4×). Output size must be statically known at ONNX export. Bilinear NOT supported on DPUCZDX8G — use nearest or replace with TransposeConv(stride=2).",
      "Resize", "B4096", True, "warning"),
    d("Resize on DPUCVDX8G/H: nearest-neighbor and bilinear both supported with fixed output sizes. Dynamic shape NOT supported.",
      "Resize", "DPUCVDX8G", True, "ok"),
    d("Resize on DPUCAHX8H/DPUCADF8H: nearest-neighbor and bilinear supported. Fixed output size required.",
      "Resize", "DPUCAHX8H", True, "ok"),

    # ═══════════════════════════════════════════════════════════════════════
    # Split
    # ═══════════════════════════════════════════════════════════════════════
    d("Split on B512/B800/B1024/B1152: NOT supported. Creates DPU subgraph boundary. Refactor C2f modules: replace torch.chunk/split with explicit Conv branches.",
      "Split", "B1024", False, "critical"),
    d("Split on B1600/B2304/B3136: NOT efficiently supported. Static equal splits cause subgraph fragmentation. Replace C2f Split with sequential Conv branches for full DPU coverage.",
      "Split", "B2304", False, "warning"),
    d("Split on B4096: static equal-split partially compiled but still fragments DPU subgraph. Replace C2f chunk operations with C3-style bottleneck to maximise DPU coverage.",
      "Split", "B4096", False, "warning"),
    d("Split on DPUCVDX8G/H: static Split with equal chunks is supported. Dynamic split sizes NOT supported.",
      "Split", "DPUCVDX8G", True, "warning"),
    d("Split on DPUCAHX8H/DPUCADF8H: static Split supported. Dynamic split NOT supported.",
      "Split", "DPUCAHX8H", True, "warning"),

    # ═══════════════════════════════════════════════════════════════════════
    # Slice
    # ═══════════════════════════════════════════════════════════════════════
    d("Slice is NOT supported on any DPUCZDX8G variant (B512–B4096). Forces CPU subgraph. Replace stem Slice with stride-2 Conv. Detection-head Slice must move to CPU post-processing.",
      "Slice", "B4096", False, "critical"),
    d("Slice is NOT supported on DPUCVDX8G/H. Move to CPU.",
      "Slice", "DPUCVDX8G", False, "critical"),
    d("Slice is NOT supported on DPUCAHX8H/DPUCADF8H. Move to CPU.",
      "Slice", "DPUCAHX8H", False, "critical"),

    # ═══════════════════════════════════════════════════════════════════════
    # Concat
    # ═══════════════════════════════════════════════════════════════════════
    d("Concat on B512: channel-axis only. Max 2 inputs. All shapes must be static. Width/height concat NOT supported.",
      "Concat", "B512", True, "warning"),
    d("Concat on B800/B1024/B1152: channel-axis only. Max 4 inputs. Static shapes required. Height/width axis NOT supported.",
      "Concat", "B1600", True, "warning"),
    d("Concat on B2304/B3136/B4096: channel-axis. Max 8 inputs. Static shapes required. FPN/PAN neck Concat supported when input resolution is fixed at export.",
      "Concat", "B4096", True, "ok"),
    d("Concat on DPUCVDX8G/H: channel-axis, no fan-in limit. Static shapes required. Width/height axis NOT supported.",
      "Concat", "DPUCVDX8G", True, "ok"),
    d("Concat on DPUCAHX8H/DPUCADF8H: channel-axis, no fan-in limit. Static shapes.",
      "Concat", "DPUCAHX8H", True, "ok"),

    # ═══════════════════════════════════════════════════════════════════════
    # BatchNormalization / Relu / LeakyRelu / Clip / HardSwish
    # ═══════════════════════════════════════════════════════════════════════
    d("BatchNormalization is fused with Conv on ALL DPU architectures. Does not appear as standalone op after Vitis AI compiler fusion. Zero overhead.",
      "BatchNormalization", "ALL", True, "ok"),
    d("Relu is natively supported on all DPU architectures (B512–B4096, DPUCVDX8G/H, DPUCAHX8H, DPUCADF8H). Zero overhead — fused with Conv+BN.",
      "Relu", "ALL", True, "ok"),
    d("LeakyRelu is NOT supported on B512 or B800. Replace with Relu on those targets. LeakyRelu IS supported on B1024, B1152, B1600, B2304, B3136, B4096, DPUCVDX8G/H, DPUCAHX8H, DPUCADF8H.",
      "LeakyRelu", "B512", False, "warning"),
    d("LeakyRelu is fully supported on B1024 and above, and all Versal/Alveo targets.",
      "LeakyRelu", "B4096", True, "ok"),
    d("Clip (ReLU6) is supported on B1024 and above. NOT supported on B512/B800.",
      "Clip", "B1024", True, "ok"),
    d("Clip (ReLU6) is NOT supported on B512 and B800.",
      "Clip", "B800", False, "warning"),
    d("HardSwish (HardSigmoid × x) is the recommended SiLU replacement. Supported on B2304, B3136, B4096, DPUCVDX8G/H, DPUCAHX8H. NOT supported on B512, B800, B1024, B1152.",
      "HardSwish", "B4096", True, "ok"),
    d("HardSwish is NOT supported on B512, B800, B1024, B1152. Use LeakyReLU(0.1) on those targets instead.",
      "HardSwish", "B1024", False, "warning"),

    # ═══════════════════════════════════════════════════════════════════════
    # Add / Mul / Sub / Div
    # ═══════════════════════════════════════════════════════════════════════
    d("Add (element-wise) for residual connections is supported on all DPU architectures. Shapes must match exactly.",
      "Add", "ALL", True, "ok"),
    d("Mul (element-wise) for scaling/attention is NOT supported on B512/B800. Forces CPU subgraph.",
      "Mul", "B512", False, "warning"),
    d("Mul (element-wise) is supported on B1024 and above for residual and channel-scaling patterns. SiLU Mul (Sigmoid×x) is NOT because Sigmoid is on CPU.",
      "Mul", "B4096", True, "ok"),
    d("Sub (element-wise) is NOT supported on B512/B800/B1024. Replace with Add + negated constant.",
      "Sub", "B1024", False, "warning"),
    d("Sub is supported on B2304 and above in limited fused patterns.",
      "Sub", "B2304", True, "warning"),
    d("Div (element-wise) is NOT supported on B512–B2304. Supported on B3136, B4096, DPUCVDX8G/H, DPUCAHX8H in specific fused patterns only.",
      "Div", "B2304", False, "warning"),
    d("Div is supported on B4096 in fused patterns (e.g. scale normalisation). Standalone Div may still fallback to CPU.",
      "Div", "B4096", True, "warning"),

    # ═══════════════════════════════════════════════════════════════════════
    # Reshape / Transpose / Flatten / Squeeze / Unsqueeze
    # ═══════════════════════════════════════════════════════════════════════
    d("Reshape with STATIC shapes is supported on all DPU architectures as a zero-cost op. Dynamic Reshape (shape inferred at runtime) is NOT supported on any DPU — causes full CPU subgraph.",
      "Reshape", "ALL", True, "warning"),
    d("Transpose at graph input/output boundaries is tolerated on B2304 and above. Mid-graph Transpose forces DPU↔CPU transition on all DPUCZDX8G variants.",
      "Transpose", "B4096", True, "warning"),
    d("Transpose is NOT supported mid-graph on B512–B1600. Move to CPU or eliminate.",
      "Transpose", "B1024", False, "warning"),
    d("Flatten is a zero-cost reshape on all architectures. Supported everywhere.",
      "Flatten", "ALL", True, "ok"),
    d("Squeeze and Unsqueeze are zero-cost with static shapes. Dynamic rank NOT supported.",
      "Squeeze", "ALL", True, "ok"),

    # ═══════════════════════════════════════════════════════════════════════
    # Gemm / MatMul
    # ═══════════════════════════════════════════════════════════════════════
    d("Gemm (FC layer) is NOT supported on B512/B800. Implement as 1×1 Conv instead.",
      "Gemm", "B512", False, "warning"),
    d("Gemm is NOT supported on B1024. Replace with 1×1 Conv.",
      "Gemm", "B1024", False, "warning"),
    d("Gemm is supported on B1152, B1600, B2304, B3136, B4096. Static input size required (fixed batch).",
      "Gemm", "B4096", True, "ok"),
    d("Gemm is fully supported on DPUCVDX8G/H and DPUCAHX8H/DPUCADF8H.",
      "Gemm", "DPUCVDX8G", True, "ok"),

    # ═══════════════════════════════════════════════════════════════════════
    # Pad / ConvTranspose / Upsample
    # ═══════════════════════════════════════════════════════════════════════
    d("Pad (zero-padding) before Conv is supported on all architectures. Asymmetric padding NOT supported on B512–B1024.",
      "Pad", "ALL", True, "ok"),
    d("ConvTranspose (deconvolution) is supported on B2304, B3136, B4096, DPUCVDX8G/H, DPUCAHX8H for 2× upsampling. NOT supported on B512, B800, B1024, B1152, B1600.",
      "ConvTranspose", "B4096", True, "ok"),
    d("ConvTranspose is NOT supported on B512, B800, B1024, B1152, B1600. Use Resize nearest 2× instead.",
      "ConvTranspose", "B1024", False, "warning"),
    d("Upsample (nearest 2×) is supported on B1024 and above as a special case of Resize.",
      "Upsample", "B1024", True, "ok"),

    # ═══════════════════════════════════════════════════════════════════════
    # Ops NOT supported on any architecture
    # ═══════════════════════════════════════════════════════════════════════
    d("NonMaxSuppression (NMS) is NOT supported on any DPU. Must run on ARM CPU as post-processing. This is standard for all detection models — expected behaviour.",
      "NonMaxSuppression", "ALL", False, "critical"),
    d("TopK is NOT supported on any DPU. Run on CPU.",
      "TopK", "ALL", False, "critical"),
    d("LSTM / GRU / RNN are NOT supported on any DPU. Run on CPU or replace with Conv1D equivalents.",
      "LSTM", "ALL", False, "critical"),
    d("LayerNormalization is NOT supported on any DPU. Replace with BatchNorm or run on CPU.",
      "LayerNormalization", "ALL", False, "critical"),
    d("Gather / GatherElements / GatherND are NOT supported on any DPU. Move to CPU.",
      "Gather", "ALL", False, "critical"),
    d("ScatterElements / ScatterND are NOT supported on any DPU.",
      "ScatterElements", "ALL", False, "critical"),
    d("Einsum is NOT supported on any DPU.",
      "Einsum", "ALL", False, "critical"),
    d("Attention (multi-head self-attention) is NOT supported on any DPU. Must run on CPU or use dedicated AI Engine on Versal.",
      "Attention", "ALL", False, "critical"),
    d("AdaptiveMaxPool is NOT supported on any DPU. Replace with fixed MaxPool.",
      "AdaptiveMaxPool", "ALL", False, "critical"),
    d("Sqrt, Exp, Log, Pow are NOT supported on any DPU. Move to CPU post-processing.",
      "Sqrt", "ALL", False, "critical"),
    d("InstanceNormalization is NOT supported on any DPU.",
      "InstanceNorm", "ALL", False, "critical"),
    d("GroupNormalization is NOT supported on any DPU.",
      "GroupNorm", "ALL", False, "critical"),
]

# ---------------------------------------------------------------------------
DPU_TARGETS = [
    "B512", "B800", "B1024", "B1152", "B1600",
    "B2304", "B3136", "B4096",
    "DPUCVDX8G", "DPUCVDX8H", "DPUCAHX8H", "DPUCADF8H",
]

BOARD_TO_ARCH = {
    "KV260":      "B4096",
    "ZCU102":     "B4096",
    "ZCU106":     "B4096",
    "Kria K26":   "B4096",
    "ZCU104":     "B2304",
    "Ultra96-v2": "B1600",
    "VCK190":     "DPUCVDX8G",
    "VEK280":     "DPUCVDX8H",
    "Alveo U50":  "DPUCAHX8H",
    "Alveo U200": "DPUCAHX8H",
    "Alveo U250": "DPUCAHX8H",
    "Alveo U280": "DPUCAHX8H",
    "Alveo U55C": "DPUCADF8H",
    "Alveo V70":  "DPUCADF8H",
}


def main():
    db_path = "./vitis_docs_db"
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
        print("Cleared existing DB.")

    print(f"Building Vitis AI compatibility vector DB ({len(VITIS_DOCS)} entries)...")
    emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    Chroma.from_documents(documents=VITIS_DOCS, embedding=emb, persist_directory=db_path)
    print(f"Done — {len(VITIS_DOCS)} entries indexed into {db_path}")
    print(f"Architectures: {', '.join(DPU_TARGETS)}")
    print(f"Board mappings: {', '.join(f'{b}→{a}' for b, a in BOARD_TO_ARCH.items())}")


if __name__ == "__main__":
    main()
