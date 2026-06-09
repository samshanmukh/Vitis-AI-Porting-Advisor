# agent_one_modelExtractor.py
import onnx
import json
import os

async def extract_model(input_data: dict) -> dict:
    """
    Agent 1: Extract model operations

    Input: {"model_path": "yolov8m.onnx"}  (or .pt — will export to ONNX first)
    Output: {"operations": [...], "total_ops": 250, "model_name": "..."}
    """

    model_path = input_data.get("model_path")
    if not model_path:
        return {"error": "model_path required"}

    try:
        ext = os.path.splitext(model_path)[1].lower()

        if ext == ".onnx":
            onnx_path = model_path
        elif ext == ".pt":
            # Use existing .onnx if already exported alongside the .pt
            onnx_path = os.path.splitext(model_path)[0] + ".onnx"
            if not os.path.exists(onnx_path):
                from ultralytics import YOLO
                YOLO(model_path).export(format="onnx", imgsz=640, opset=12)
        else:
            return {"error": f"Unsupported file type: {ext}. Pass a .onnx or .pt file."}

        onnx_model = onnx.load(onnx_path)
        ops = []

        for node in onnx_model.graph.node:
            ops.append({
                "name": node.name,
                "type": node.op_type,
                "inputs": list(node.input),
                "outputs": list(node.output),
            })

        return {
            "model_name": model_path,
            "total_ops": len(ops),
            "operations": ops,
            "status": "success"
        }

    except Exception as e:
        return {
            "error": str(e),
            "model_name": model_path
        }

# For local testing
if __name__ == "__main__":
    import asyncio
    result = asyncio.run(extract_model({"model_path": "yolov8m.onnx"}))
    print(json.dumps(result, indent=2))