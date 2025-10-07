# VERSÃO ULTRA SIMPLES - processamento individual
import pandas as pd
from PIL import Image
from datasets import Dataset
from transformers import BlipProcessor, BlipForConditionalGeneration, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import torch

# Config
CSV_PATH = "../data/final_embed448_copy_norep.csv"
MODEL_ID = "Salesforce/blip-image-captioning-base"
OUTPUT_DIR = "./blip-working"

def normalize_label(desc):
    if not isinstance(desc, str): return None
    desc = desc.lower().strip()
    if "cranio-caudal exaggerated" in desc: return "XCC"
    elif "cranio-caudal" in desc: return "CC"
    elif "medio-lateral oblique" in desc: return "MLO"
    elif "medio-lateral" in desc: return "ML"
    return None

# Carregar dados
df = pd.read_csv(CSV_PATH, dtype=str)
df['view'] = df["0_ViewCodeSequence_CodeMeaning"].apply(normalize_label)
df = df[df['view'].notna()].reset_index(drop=True)
print(f"📊 {len(df)} amostras")

# Carregar modelo
processor = BlipProcessor.from_pretrained(MODEL_ID)
model = BlipForConditionalGeneration.from_pretrained(MODEL_ID)

# LoRA
lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["query", "value"], lora_dropout=0.05)
model = get_peft_model(model, lora_config)

# PRÉ-PROCESSAMENTO INDIVIDUAL SIMPLES
print("🔄 Pré-processando...")

processed_samples = []
for idx, row in df.iterrows():
    try:
        image = Image.open(row['path']).convert('RGB')
        inputs = processor(
            images=image,
            text=row['view'],
            padding="max_length",
            max_length=32,
            truncation=True,
            return_tensors="pt"
        )
        
        # Garantir shapes corretos
        processed_samples.append({
            'pixel_values': inputs.pixel_values[0],  # Remove batch dimension
            'input_ids': inputs.input_ids[0],
            'attention_mask': inputs.attention_mask[0],
            'labels': inputs.input_ids[0].clone()  # Labels = input_ids para causal LM
        })
        
        if idx % 10 == 0:
            print(f"✅ {idx+1}/{len(df)}")
            
    except Exception as e:
        print(f"❌ Erro em {row['path']}: {e}")
        continue

# Criar dataset
if processed_samples:
    dataset = Dataset.from_list(processed_samples)
    print(f"🎉 Dataset: {len(dataset)} amostras")
    
    # Treinar
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        learning_rate=1e-4,
        fp16=True,
        logging_steps=10,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=processor.tokenizer,
    )
    
    print("🚀 Treinando...")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    print(f"💾 Salvo em: {OUTPUT_DIR}")
