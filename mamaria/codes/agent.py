import json
from typing import Any, Dict, TypedDict, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import Ollama
from langgraph.graph import StateGraph, START, END

class State(TypedDict, total=False):
    mri_summary: str
    clip_attention: List[Dict[str, Any]]
    clip_summary: List[Dict[str, Any]]
    llm_response_raw: str
    llm_response: Dict[str, Any]
    final_report: Dict[str, Any]

def summarize_clip_attention(attns, top_k=3):
    sorted_by_score = sorted(attns, key=lambda x: x.get('score', 0), reverse=True)
    return sorted_by_score[:top_k]

def prepare_inputs(state: State):
    clip_attention = state.get('clip_attention', [])
    clip_summary = state.get('clip_summary', [])
    mri_summary = state.get('mri_summary', 'MRI: unknown; provide details.')
    return {'mri_summary': mri_summary, 'clip_attention': clip_attention, 'clip_summary': clip_summary}

model = Ollama(model='llama3', temperature=0)

PROMPT = """
SYSTEM: You are a research assistant helping a radiologist (role: evidence_analyst).
You are given the following inputs:
INPUT_MRI: {mri_summary}
INPUT_CLIP: {clip_summary}
INSTRUCTION: Based on both inputs, assign this JSON exactly as follows:
{
  "cancer": true | false,
  "confidence": float (0.0 to 1.0),
  "notes": "string with details and reasoning"
}
Make sure to only respond with valid JSON. If you are unsure, respond with "cancer": false and "confidence": 0.0.
"""

prompt = ChatPromptTemplate.from_template(PROMPT)

def llm_eval(state: State):
    mri_summary = state.get('mri_summary')
    clip_summary = state.get('clip_summary')
    chain = prompt | model
    llm_out = chain.invoke({'mri_summary': mri_summary, 'clip_summary': clip_summary})
    try:
        result_text = llm_out['text'] if isinstance(llm_out, dict) else llm_out
        parsed = json.loads(result_text)
    except Exception as e:
        parsed = {'raw': llm_out, 'error': str(e)}
    return {'llm_response_raw': llm_out, 'llm_response': parsed}

def final_report(state: State):
    return {'final_report': state.get('llm_response')}

builder = StateGraph(State)
builder.add_node('prepare_inputs', prepare_inputs)
builder.add_node('llm_eval', llm_eval)
builder.add_node('final_report', final_report)
builder.add_edge(START, 'prepare_inputs')
builder.add_edge('prepare_inputs', 'llm_eval')
builder.add_edge('llm_eval', 'final_report')
builder.add_edge('final_report', END)
builder.set_entry_point('prepare_inputs')
graph = builder.build()

example_state: State = {
    'mri_summary': 'Breast MRI, T1 post-contrast, slice_thickness = 1mm',
    'clip_attention': [
        {'coords': [100, 120, 40, 45], 'score': 0.92, 'token': 'mass'},
        {'coords': [200, 220, 30, 30], 'score': 0.34, 'token': 'skin'},
        {'coords': [60, 200, 25, 25], 'score': 0.12, 'token': 'artifact'}
    ]
}

result = graph.invoke(example_state)
print(json.dumps(result, indent=2, ensure_ascii=False))
