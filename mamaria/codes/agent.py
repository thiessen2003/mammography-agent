import pandas as pd
import base64
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.chat_models import ChatOllama
import json

df = pd.read_csv("../data/final_embed448_copy.csv")

llava = ChatOllama(
    model="llava",
    format="json",
    temperature=0.0
)

system_msg = SystemMessage(content="""
You are an expert radiologist in breast cancer.
You will receive patient's textual metadata and a breast MRI image.
Your task: identify abnormalities in the image and/or textual data, analyze them, and diagnose.
Respond strictly in JSON with these fields:
- TUMOR: "YES" or "NO"
- CERTAINTY: float between 0.0 and 1.0
- TUMOR_LOCATION: string or null
- NOTES: free text
""".strip())

results = []

for _, row in df.iterrows():
    image_path = row["path"]
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    image_content = f"data:image/jpeg;base64,{image_b64}"  # adjust MIME type if PNG etc.

    metadata = row.drop(labels=["path"]).to_dict()

    user_msgs = [
        HumanMessage(content=[
            {"type": "text", "text": f"Textual data: {metadata}"},
            {"type": "image_url", "image_url": image_content}
        ])
    ]

    response = llava.invoke([system_msg] + user_msgs)

    parsed = json.loads(response.content)

    results.append(parsed)

df["diagnosis"] = results
df.to_csv("../data/final_embed_448_with_reports.csv", index=False)

