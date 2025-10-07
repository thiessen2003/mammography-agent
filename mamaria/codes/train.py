# ULTRA SIMPLE VERSION - individual processing
import pandas as pd
from PIL import Image
from datasets import Dataset
from transformers import BlipProcessor, BlipForConditionalGeneration, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import torch
import json

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

def create_training_target(view):
    """Create training targets with pre-defined explanations"""
    view_names = {
        "CC": "craniocaudal (CC)",
        "XCC": "exaggerated craniocaudal (XCC)", 
        "ML": "mediolateral (ML)",
        "MLO": "mediolateral oblique (MLO)"
    }
    
    # Simple pre-defined explanations for training only
    explanations = {
        "CC": "standard top-to-bottom compression view",
        "XCC": "extended craniocaudal view for lateral tissue",
        "ML": "pure lateral mediolateral view", 
        "MLO": "oblique view including axilla"
    }
    
    target_data = {
        "view": view_names.get(view, "undetermined"),
        "explanation": explanations.get(view, "view classification")
    }
    return json.dumps(target_data, ensure_ascii=False)

# Load data
df = pd.read_csv(CSV_PATH, dtype=str)
df['view'] = df["0_ViewCodeSequence_CodeMeaning"].apply(normalize_label)
df = df[df['view'].notna()].reset_index(drop=True)
print(f"Data: {len(df)} samples")

# Load model
processor = BlipProcessor.from_pretrained(MODEL_ID)
model = BlipForConditionalGeneration.from_pretrained(MODEL_ID)

# LoRA
lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["query", "value"], lora_dropout=0.05)
model = get_peft_model(model, lora_config)

# SIMPLE INDIVIDUAL PRE-PROCESSING
print("Pre-processing...")

processed_samples = []
for idx, row in df.iterrows():
    try:
        image = Image.open(row['path']).convert('RGB')
        
        # Create target text in JSON format (with pre-defined explanations for training)
        target_text = create_training_target(row['view'])
        
        inputs = processor(
            images=image,
            text=target_text,  # Using pre-tagged JSON as training target
            padding="max_length",
            max_length=64,  # Increased for JSON
            truncation=True,
            return_tensors="pt"
        )
        
        # Ensure correct shapes
        processed_samples.append({
            'pixel_values': inputs.pixel_values[0],  # Remove batch dimension
            'input_ids': inputs.input_ids[0],
            'attention_mask': inputs.attention_mask[0],
            'labels': inputs.input_ids[0].clone()  # Labels = input_ids for causal LM
        })
        
        if idx % 10 == 0:
            print(f"Processed {idx+1}/{len(df)}")
            
    except Exception as e:
        print(f"Error in {row['path']}: {e}")
        continue

# Create dataset
if processed_samples:
    dataset = Dataset.from_list(processed_samples)
    print(f"Dataset: {len(dataset)} samples")
    
    # Training
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        learning_rate=1e-4,
        fp16=True,
        logging_steps=10,
        prediction_loss_only=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=processor.tokenizer,
    )
    
    print("Training...")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    print(f"Saved to: {OUTPUT_DIR}")

# Function for post-training inference - model generates explanations freely
def generate_prediction(model, processor, image_path):
    """Generate prediction with free-form explanations after training"""
    try:
        image = Image.open(image_path).convert('RGB')
        
        # Generate caption - model will create its own explanations
        inputs = processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=64,
                num_beams=5,
                early_stopping=True,
                do_sample=True,  # Allow creativity in explanations
                temperature=0.7,  # Balance between creativity and consistency
            )
        
        prediction = processor.decode(outputs[0], skip_special_tokens=True)
        
        # Try to parse as JSON
        try:
            result = json.loads(prediction)
            return result
        except json.JSONDecodeError:
            # If not valid JSON, try to extract view and explanation
            return {"raw_output": prediction}
            
    except Exception as e:
        return {"error": str(e)}

# Function to test multiple images
def batch_predict(model, processor, image_paths):
    """Generate predictions for multiple images"""
    results = []
    for image_path in image_paths:
        result = generate_prediction(model, processor, image_path)
        results.append({
            "image": image_path,
            "prediction": result
        })
    return results

print("\nTraining completed! The model can now generate free-form explanations.")
print("Use generate_prediction() for single images or batch_predict() for multiple images.")
print("The model will create its own explanations based on learned patterns.")
