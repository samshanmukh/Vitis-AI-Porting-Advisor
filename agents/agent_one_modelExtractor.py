# agent_one_modelExtractor.py
import torch
import onnx
import json

async def extract_model(input_data: dict) -> dict:
    """
    RocketRide agent: Extract model operations
    
    Input: {"model_path": "yolov8m.pt"}
    Output: {"operations": [...], "total_ops": 250, "model_name": "..."}
    """
    
    model_path = input_data.get("model_path")
    if not model_path:
        return {"error": "model_path required"}
    
    try:
        # Your existing extraction logic
        model = torch.load(model_path)
        dummy_input = torch.randn(1, 3, 224, 224)
        onnx_path = "temp.onnx"
        torch.onnx.export(model, dummy_input, onnx_path)
        
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
    result = asyncio.run(extract_model({"model_path": "yolov8m.pt"}))
    print(json.dumps(result, indent=2))