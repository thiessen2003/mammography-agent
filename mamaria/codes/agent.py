import json
import base64
from typing import Any, Dict, List, TypedDict, Optional
from dataclasses import dataclass
from PIL import Image
import io

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import Ollama
from langchain_ollama.chat_models import ChatOllama
from langgraph.graph import StateGraph, START, END

class State(TypedDict, total=False):
    mri_image: str
    attention_heatmap: str
    prepared_inputs: Dict[str, Any]
    llm_raw: Any
    llm_parsed: Dict[str, Any]
    final_report: Dict[str, Any]

def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def prepare_multimodal_inputs(state: State) -> Dict[str, Any]:
    mri_image = state.get("mri_image", "")
    heatmap_image = state.get("attention_heatmap", "")
    
    prepared = {
        "mri_image": mri_image,
        "heatmap_image": heatmap_image,
    }
    state["prepared_inputs"] = prepared
    return state

MULTIMODAL_PROMPT = """
SYSTEM: You are an evidence_analyst helping a radiologist analyze breast medical images.
You will be given two images:
1. A breast MRI image
2. A heatmap highlighting dense regions of interest in the breast

INSTRUCTION: Analyze both images carefully and output ONLY a single JSON object (no surrounding text) with the following exact schema:

{
  "dense_tissue_present": true | false,
  "density_confidence": <float between 0.0 and 1.0>,
  "dense_regions_locations": <list of strings describing locations or null>,
  "birads_score": <string like "BI-RADS 1-6">,
  "recommended_next_step": <string>,
  "clinical_notes": <string with concise reasoning about tissue density and patterns>
}

If you are unsure or images are unclear, set "dense_tissue_present": false and "density_confidence": 0.0 and explain in clinical_notes.
"""

multimodal_model = ChatOllama(
    model="llava",
    temperature=0.0,
    format="json"
)

def multimodal_eval(state: State) -> Dict[str, Any]:
    prepared = state.get("prepared_inputs", {})
    mri_image = prepared.get("mri_image", "")
    heatmap_image = prepared.get("heatmap_image", "")
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": MULTIMODAL_PROMPT},
                {"type": "image", "image": mri_image},
                {"type": "image", "image": heatmap_image}
            ]
        }
    ]
    
    try:
        response = multimodal_model.invoke(messages)
        raw_text = response.content if hasattr(response, 'content') else str(response)
        
        parsed: Dict[str, Any]
        try:
            parsed = json.loads(raw_text)
            required_keys = ["dense_tissue_present", "density_confidence", "birads_score"]
            if not all(key in parsed for key in required_keys):
                raise ValueError("Parsed JSON missing required keys.")
        except json.JSONDecodeError:
            parsed = {
                "dense_tissue_present": False,
                "density_confidence": 0.0,
                "dense_regions_locations": None,
                "birads_score": "BI-RADS 0",
                "recommended_next_step": "Unable to parse LLM output. Review images and try again.",
                "clinical_notes": f"JSON parsing failed. Raw output: {str(raw_text)[:500]}"
            }
    except Exception as e:
        parsed = {
            "dense_tissue_present": False,
            "density_confidence": 0.0,
            "dense_regions_locations": None,
            "birads_score": "BI-RADS 0",
            "recommended_next_step": "LLM invocation failed. Check model availability.",
            "clinical_notes": f"Error: {str(e)}"
        }
        raw_text = f"Error: {str(e)}"
    
    state["llm_raw"] = raw_text
    state["llm_parsed"] = parsed
    return state

def final_multimodal_report(state: State) -> Dict[str, Any]:
    parsed = state.get("llm_parsed", {})
    report = {
        "density_assessment": parsed,
        "llm_raw": state.get("llm_raw"),
        "meta": {
            "model_used": "llava",
            "notes": "This is an assistive research output for breast density assessment; validate clinically."
        }
    }
    state["final_report"] = report
    return state

builder = StateGraph(State)
builder.add_node("prepare_inputs", prepare_multimodal_inputs)
builder.add_node("multimodal_eval", multimodal_eval)
builder.add_node("final_report", final_multimodal_report)
builder.add_edge(START, "prepare_inputs")
builder.add_edge("prepare_inputs", "multimodal_eval")
builder.add_edge("multimodal_eval", "final_report")
builder.add_edge("final_report", END)
builder.set_entry_point("prepare_inputs")
graph = builder.build()

if __name__ == "__main__":
    example_state: State = {
        "mri_image": "base64_encoded_mri_image_here",
        "attention_heatmap": "base64_encoded_heatmap_here"
    }
    
    out = graph.invoke(example_state)
    print(json.dumps(out["final_report"], indent=2, ensure_ascii=False))
