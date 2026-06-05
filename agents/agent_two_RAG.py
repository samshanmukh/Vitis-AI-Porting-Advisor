# agent_two_RAG.py
import json
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# Load Vitis docs once at startup
try:
    embeddings = OpenAIEmbeddings()
    vector_db = Chroma(embedding_function=embeddings, persist_directory="./vitis_docs_db")
except:
    vector_db = None
    print("Warning: Vitis docs DB not found. Run setup_vitis_rag.py first.")

async def search_vitis_compatibility(input_data: dict) -> dict:
    """
    RocketRide agent: Cross-reference ops against Vitis AI docs
    
    Input: {"operations": [{"type": "Conv", ...}, ...]}
    Output: {"compatibility_report": [...], "unsupported_ops": [...]}
    """
    
    operations = input_data.get("operations", [])
    if not operations:
        return {"error": "operations list required"}
    
    if not vector_db:
        return {"error": "Vitis documentation not loaded"}
    
    compatibility_report = []
    unsupported_ops = []
    
    # Check each operation
    for op in operations:
        op_type = op.get("type")
        query = f"Is {op_type} supported in Vitis AI?"
        
        try:
            # Search Vitis docs
            results = vector_db.similarity_search(query, k=1)
            doc_content = results[0].page_content if results else "No documentation found"
            
            is_supported = "supported" in doc_content.lower() and "not" not in doc_content.lower()
            
            compatibility_report.append({
                "op_type": op_type,
                "supported": is_supported,
                "vitis_info": doc_content[:300],  # First 300 chars
                "op_name": op.get("name")
            })
            
            if not is_supported:
                unsupported_ops.append({
                    "name": op.get("name"),
                    "type": op_type
                })
        
        except Exception as e:
            compatibility_report.append({
                "op_type": op_type,
                "error": str(e)
            })
    
    return {
        "compatibility_report": compatibility_report,
        "unsupported_ops": unsupported_ops,
        "total_checked": len(operations),
        "unsupported_count": len(unsupported_ops),
        "status": "success"
    }

# For local testing
if __name__ == "__main__":
    import asyncio
    test_ops = [
        {"name": "Conv_0", "type": "Conv"},
        {"name": "Pool_0", "type": "AdaptiveAvgPool"}
    ]
    result = asyncio.run(search_vitis_compatibility({"operations": test_ops}))
    print(json.dumps(result, indent=2))