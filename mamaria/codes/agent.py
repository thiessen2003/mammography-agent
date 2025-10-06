import pandas as pd
import base64
from PIL import Image
from io import BytesIO
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama.chat_models import ChatOllama
import json

CSV_PATH = '../data/final_embed448_copy_norep.csv'

df = pd.read_csv(CSV_PATH)

llava = ChatOllama(
        model = 'rohithbojja/llava-med-v1.6',
        format = 'json',
        temperature = 0
        )

system_msg = SystemMessage(content = """
                           You are an expert radiologist in breast cancer and digital mammography.

                           CLASSIFICATION RULES:

                           - Never guess blindly.
                           - For every input, choose exactly one of the four breast views, unless there is unsufficient information.
                           - Follow the decision checklist before answering.
                           - If the description is ambiguous, incomplete, or does not match any view with confidence, then output "undetermined".
                           - Output ONLY valid JSON. No extra text, no markdown, no explanations outside the JSON.

                           TASK:
                           - Identify the mammographic view from the following list:
                           1. craniocaudal (CC)
                           2. exaggerated craniocaudal (XCC)
                           3. mediolateral (ML)
                           4. mediolateral oblique (MLO)
                           5. undetermined (fallback option when input is unclear)

                           OUTPUT:
                           - Answer strictly in this JSON format:
                           {
                            "view": "<breast view or 'undetermined'>",
                            "explanation": "<one-sentence justification>"
                           }

                           --------------------------------------------------------

                           DECISION CHECKLIST (always apply in order):

                           1. *Compression direction*
                           - If top-to-bottom -> candidate is CC or XCC.
                           - If side-to-side -> candidate is ML or MLO.

                           2. *Beam orientation*
                           - If strictly vertical (straight down) -> CC.
                           - If vertical with tissue pulled laterally/medially -> XCC.
                           - If strictly horizontal (pure lateral profile) -> ML.
                           - If diagonal (30 - 60 degree oblique, includes axilla) -> MLO.

                           3. *Rule out errors*
                           - CC is NOT shifted laterally (otherwise XCC).
                           - XCC is NOT pure CC, and NOT angled (MLO).
                           - ML is NOT angled (MLO) and NOT vertical (CC/XCC).
                           - MLO is NOT strictly horizontal (ML) and NOT vertical (CC/XCC).

                           4. *Fallback*
                           - If the input lacks compression direction or angle cues -> "undetermined".

                           -------------------------------------------------------

                           BREAST VIEW DEFINITIONS WITH NEGATIVE COUNTEREXAMPLES:

                           1. Exaggerated Craniocaudal (XCC):
                           - Compression top-to-bottom (like CC).
                           - Breast pulled laterally or medially to include peripheral tissue not seen in standard CC.
                           - NOT: a standard CC without extension; NOT: side-to-side or angle views.

                           2. Craniocaudal (CC):
                           - Compression top-to-bottom.
                           - X-ray beam vertical.
                           - Shows medial and lateral breast tissue in a straight top-down view.
                           - NOT: shifted laterally or medially (that would be XCC); NOT: angled or oblique.

                           3. Mediolateral (ML):
                           - Compression medial-to-lateral.
                           - X-ray beam horizontal.
                           - Produces a pure lateral profile.
                           - NOT: angle oblique (that would be MLO); NOT: top-down (CC/XCC).

                           4. Mediolateral Oblique (MLO):
                           - Compression medial-to-lateral at an oblique angle (30 - 60 degrees).
                           - X-ray beam diagonal.
                           - Shows upper outer quadrant and axilla.
                           - NOT: strictly horizontal (ML); NOT: vertical (CC/XCC).

                           ------------------------------------------------------

                           CONTRAST SUMMARY (memorize for classification):
                           
                           - CC = vertical top-down.
                           - XCC = vertical top-down, but shifted sideways to include extra tissue.
                           - ML = pure side-to-side (horizontal).
                           - MLO = angled diagonal (30 - 60 degrees), includes axilla.

                           ------------------------------------------------------

                           EXAMPLES:

                           Input: "Digital mammography. Breast compressed top-to-bottom, image emphasizes peripheral tissue."
                           Output: 
                           {
                            "view": "exaggerated craniocaudal (XCC)",
                            "explanation": "top-down compression with lateral extension beyond standard CC."
                           }

                           Input: "Digital mammography. Breast compressed top-to-bottom, image shows medial and lateral regions equally."
                           Output:
                           {
                            "view": "craniocaudal (CC)",
                            "explanation": "straight vertical compression showing medial and lateral tissue."
                           }

                           Input: "Digital mammography. Breast compressed side-to-side, producing a pure lateral profile."
                           Output:
                           {
                            "view": "mediolateral (ML)",
                            "explanation": "horizontal compression creates strict side profile."
                           } 

                           input: "Digital mammography. Breast compressed at an oblique angle, including axilla."
                           Output:
                           {
                            "view": "mediolateral oblique (MLO)",
                            "explanation": "diagonal orientation highlights axilla and upper outer quadrant."
                           }

                           Input: "Digital mammography. Breast image with no clear compression direction or angle."
                           Output:
                           {
                            "view": "undetermined",
                            "explanation": "insufficient information about compression direction or orientation."
                           } 
                           """.strip())

results = []

for i, row in df.iterrows():
    if i > 20:
        break

    image_path = row['path']
    
    print(f"Image path for {i}-th case: {image_path}")
    img = Image.open(image_path).convert('RGB')
    buffer = BytesIO()
    img.save(buffer, format = "JPEG")
    image_bytes = buffer.getvalue()
    image_b64 = base64.b64encode(image_bytes).decode('utf-8')
    image_content = f"data:image/jpg;base64,{image_b64}"


    user_msgs = [
            HumanMessage(content=[
                {'type': 'text', 'text': f"This is a digital mammography. From which view has it been taken?"}, 
                {'type': 'image_url', 'image_url': image_content}
                ])
            ]

    response = llava.invoke([system_msg] + user_msgs)

    parsed = response.content

    results.append(parsed)

    print(f"For {i}-th case: {parsed}")
