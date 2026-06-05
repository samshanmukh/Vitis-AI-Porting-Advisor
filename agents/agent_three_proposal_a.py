# agent_three_proposal_a.py
from anthropic import Anthropic

client = Anthropic()

async def generate_conservative_proposal(input_data: dict) -> dict:
    """
    RocketRide agent: Conservative refactoring proposal
    
    Input: {
        "model_name": "yolov8m",
        "operations": [...],
        "unsupported_ops": [...]
    }
    Output: {
        "strategy": "...",
        "changes": [...],
        "estimated_accuracy_drop": "...",
        "dev_effort": "..."
    }
    """
    
    model_name = input_data.get("model_name", "unknown")
    unsupported_ops = input_data.get("unsupported_ops", [])
    
    if not unsupported_ops:
        return {
            "strategy": "No changes needed",
            "changes": [],
            "compatibility": "100%"
        }
    
    # Build prompt for Claude
    unsupported_list = "\n".join([f"  - {op['name']} ({op['type']})" for op in unsupported_ops])
    
    prompt = f"""
You are an AI expert on Vitis AI hardware acceleration for PyTorch models.

Model: {model_name}
Unsupported operations:
{unsupported_list}

Generate a CONSERVATIVE refactoring strategy:
- Minimize code changes
- Accept some ops remaining unsupported
- Focus on quick, low-risk replacements

Return ONLY valid JSON (no markdown, no explanation) with this structure:
{{
  "strategy": "brief description",
  "changes": [
    {{"op_name": "...", "current": "...", "replacement": "...", "reason": "..."}}
  ],
  "estimated_accuracy_drop": "0.5-1.2%",
  "vitis_compatibility": "85%",
  "dev_effort": "2-4 hours"
}}
"""
    
    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1000,
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

if __name__ == "__main__":
    import asyncio
    import json
    test_input = {
        "model_name": "yolov8m",
        "unsupported_ops": [
            {"name": "Pool_5", "type": "AdaptiveAvgPool"},
            {"name": "Pool_8", "type": "AdaptiveAvgPool"}
        ]
    }
    result = asyncio.run(generate_conservative_proposal(test_input))
    print(json.dumps(result, indent=2))