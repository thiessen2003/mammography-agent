# ==========================================
# PREPROCESSAMENTO + FINE-TUNING (LLAVA-MED)
# ==========================================

import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq, Trainer, TrainingArguments

# -------------------------------------
# CONFIGURAÇÕES
# -------------------------------------
CSV_PATH = '../data/final_embed448_copy_norep.csv'
IMAGE_SIZE = (336, 1334)       # dimensões esperadas pelo modelo
MODEL_ID = "rohithbojja/llava-med-v1.6"
OUTPUT_DIR = "./llava-med-finetuned"

# -------------------------------------
# NORMALIZAÇÃO DE LABELS
# -------------------------------------
def normalize_label(desc: str) -> str | None:
    if not isinstance(desc, str):
        return None

    desc = desc.lower().strip()

    if "cranio-caudal exaggerated" in desc or "cranio-caudal exaggerated laterally" in desc:
        return "exaggerated craniocaudal (XCC)"
    elif "cranio-caudal" in desc:
        return "craniocaudal (CC)"
    elif "medio-lateral oblique" in desc or "mediolateral oblique" in desc:
        return "mediolateral oblique (MLO)"
    elif "medio-lateral" in desc or "mediolateral" in desc:
        return "mediolateral (ML)"
    else:
        return None


# -------------------------------------
# CARREGAR E PREPARAR DADOS
# -------------------------------------
df = pd.read_csv(CSV_PATH)

if "0_ViewCodeSequence_CodeMeaning" not in df.columns or "path" not in df.columns:
    raise ValueError("O CSV deve conter as colunas '0_ViewCodeSequence_CodeMeaning' e 'path'.")

df["view"] = df["0_ViewCodeSequence_CodeMeaning"].apply(normalize_label)
df = df[df["view"].notna()].reset_index(drop=True)

print("Distribuição das labels normalizadas:")
print(df["view"].value_counts(), "\n")

# -------------------------------------
# REDIMENSIONAR IMAGENS
# -------------------------------------
def resize_image(image_path: str, size: tuple[int, int] = IMAGE_SIZE):
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img = img.resize(size)
            img.save(image_path)
    except Exception as e:
        print(f"Erro ao processar {image_path}: {e}")

for img_path in df["path"]:
    if os.path.exists(img_path):
        resize_image(img_path)
    else:
        print(f"Imagem não encontrada: {img_path}")

# -------------------------------------
# DIVISÃO TREINO/VALIDAÇÃO
# -------------------------------------
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["view"], random_state=42)
train_df.to_csv("train_clean.csv", index=False)
val_df.to_csv("val_clean.csv", index=False)

print(f"Treino: {len(train_df)} imagens | Validação: {len(val_df)} imagens\n")


# -------------------------------------
# DEFINIÇÃO DO DATASET
# -------------------------------------
class MammographyDataset(Dataset):
    def __init__(self, df, image_dir, processor):
        self.df = df.reset_index(drop=True)
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row["path"]
        image = Image.open(image_path).convert("RGB")

        prompt = f"Identify the mammographic view of this image."
        text = row["view"]

        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
            padding=True
        )

        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["labels"] = self.processor.tokenizer(text, return_tensors="pt").input_ids.squeeze(0)

        return inputs


# -------------------------------------
# PREPARAR MODELO E PROCESSADOR
# -------------------------------------
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForVision2Seq.from_pretrained(MODEL_ID)

train_dataset = MammographyDataset(train_df, IMAGE_DIR, processor)
val_dataset = MammographyDataset(val_df, IMAGE_DIR, processor)

# -------------------------------------
# CONFIGURAÇÃO DO TREINAMENTO
# -------------------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    logging_dir="./logs",
    logging_steps=50,
    remove_unused_columns=False,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# -------------------------------------
# INÍCIO DO TREINAMENTO
# -------------------------------------
trainer.train()

print("\nTreinamento finalizado com sucesso.")
print(f"Modelo salvo em: {OUTPUT_DIR}")

