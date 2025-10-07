from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from PIL import Image
import json

OUTPUT_DIR = "./blip-working"

PROMPT = """
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
                           {view: <breast view or undetermined>, explanation: <one-sentence justification>}

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
                           - If the input lacks compression direction or angle cues -> undetermined.
"""

# Recarrega modelo e processor
processor = BlipProcessor.from_pretrained(OUTPUT_DIR)
model = BlipForConditionalGeneration.from_pretrained(OUTPUT_DIR)
model.eval().to("cuda" if torch.cuda.is_available() else "cpu")

def generate_prediction(model, processor, image_path):
    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, text=PROMPT, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=64,
            num_beams=5,
            early_stopping=True,
            do_sample=True,
            temperature=0.7
        )

    text = processor.decode(outputs[0], skip_special_tokens=True)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw_output": text}

# Exemplo
result = generate_prediction(model, processor, "/mnt/d/Users/miguel/embed336x1334/60790166_3323267960938479_cranio-caudal_L.jpg")
print(result)
