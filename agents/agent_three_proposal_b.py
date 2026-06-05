# agent_three_proposal_b.py
from anthropic import Anthropic
import json

client = Anthropic()

async def generate_aggressive_proposal(input_data: dict) -> dict:
    """
    RocketRide agent: Aggressive refactoring + quantization
    
    Input: same as proposal_a
    Output: Full refactoring with INT8 quantization strategy
    """
    
    model_name = input_data.get("model_name", "unknown")
    unsupported_ops = input_data.get("unsupported_ops", [])
    
    unsupported_list = "\n".join([f"  - {op['name']} ({op['type']})" for op in unsupported_ops])
    
    prompt = f"""
You are an AI expert on Vitis AI hardware acceleration for PyTorch models.

Model: {model_name}
Unsupported operations:
{unsupported_list}

Generate an AGGRESSIVE refactoring strategy:
- Achieve 100% Vitis AI compatibility
- Include INT8 quantization for hardware acceleration
- Provide performance estimates
- Generate deployment-ready approach

Return ONLY valid JSON (no markdown) with this structure:
{{
  "strategy": "Full refactoring with INT8 quantization",
  "changes": [
    {{"op_name": "...", "current": "...", "replacement": "...", "reason": "..."}}
  ],
  "quantization_strategy": "Post-training quantization (PTQ) with calibration",
  "estimated_accuracy_drop": "0.2-0.8%",
  "vitis_compatibility": "100%",
  "inference_speedup": "3.5-4.2x on Xilinx hardware",
  "dev_effort": "6-10 hours"
}}
"""
    
    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1200,
        messages=[{"role": "user", "content": prompt}]
    )
    
    try:
        proposal = json.loads(message.content[0].text)
        proposal["status"] = "success"
        return proposal
    except json.JSONDecodeError:
        return {
            "error": "Failed to parse Claude response",
            "raw_response": message.content[0].text
        }