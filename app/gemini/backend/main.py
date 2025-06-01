import os
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load API key from .env
load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("Gemini API key not found in .env")

# Configure Gemini API
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Gemini model
MODEL_NAME = "models/gemini-1.5-flash"  # or use "models/gemini-1.5-pro"
model = genai.GenerativeModel(MODEL_NAME)

# Initialize Transformer model
BERT_MODEL_NAME = "cahya/bert2bert-indonesian-summarization"
tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
bert_model = AutoModelForSeq2SeqLM.from_pretrained(BERT_MODEL_NAME)

# Load tuning.py model at startup
TUNING_MODEL_DIR = "models/best_model_epoch_2"
tuning_tokenizer, tuning_model, tuning_device = None, None, None

def load_tuning_model():
    global tuning_tokenizer, tuning_model, tuning_device
    tuning_tokenizer = AutoTokenizer.from_pretrained(TUNING_MODEL_DIR)
    tuning_model = AutoModelForSeq2SeqLM.from_pretrained(TUNING_MODEL_DIR)
    tuning_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tuning_model.to(tuning_device)

load_tuning_model()

def tuning_generate_headline(article):
    tuning_model.eval()
    inputs = tuning_tokenizer(
        article,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors="pt"
    ).to(tuning_device)
    with torch.no_grad():
        outputs = tuning_model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=128,
            num_beams=4,
            early_stopping=True
        )
    headline = tuning_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return headline

# Initialize FastAPI
app = FastAPI(title="Article Title Generator API")

# Request data schema
class TitleRequest(BaseModel):
    content: str

# Function to build prompt for title generation
def build_prompt(content: str) -> str:
    return f"""Tolong buatkan judul yang menarik dalam Bahasa Indonesia untuk konten artikel berikut ini.
Judul harus memenuhi kriteria berikut:
- Singkat, padat, dan menarik perhatian
- Mencerminkan topik utama artikel dengan akurat
- Menggunakan kalimat aktif, bukan pasif
- Panjang antara 5-12 kata
- Tidak clickbait, tapi tetap membuat penasaran
- Bahasa Indonesia yang baik dan benar
- Jika ada istilah teknis, pastikan sesuai konteks Indonesia

Contoh Judul yang Baik:
- "Strategi Efektif Mengelola Keuangan Pribadi di Era Digital"
- "Teknologi AI: Peluang dan Tantangan di Dunia Kerja Modern"
Konten artikel:
{content}

Berikan hanya judulnya saja tanpa penjelasan tambahan atau tanda kutip."""

# Endpoint for title generation
@app.post("/generate/")
async def generate_title(req: TitleRequest):
    if len(req.content.strip()) < 50:
        raise HTTPException(status_code=400, detail="Article content is too short. Please provide more content.")

    try:
        results = {}
        
        # Generate title using Gemini
        prompt = build_prompt(req.content)
        response = model.generate_content(prompt)
        gemini_title = response.text.strip()
        
        # Generate title using BERT model
        inputs = tokenizer(req.content, return_tensors="pt", max_length=512, truncation=True)
        summary_ids = bert_model.generate(
            inputs["input_ids"],
            max_length=40,
            min_length=5,
            num_beams=4,
            early_stopping=True
        )
        bert_title = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        # Generate title using tuning.py model
        tuning_title = tuning_generate_headline(req.content)
        if not tuning_title:
            raise HTTPException(status_code=500, detail="Failed to generate tuning.py title")
        tuning_metrics = {
            "length": len(tuning_title.split()),
            "character_count": len(tuning_title)
        }

        # Validate results
        if not gemini_title or not bert_title or not tuning_title:
            raise HTTPException(status_code=500, detail="Failed to generate one or more titles")

        # Calculate metrics for all titles
        gemini_metrics = {
            "length": len(gemini_title.split()),
            "character_count": len(gemini_title)
        }
        
        bert_metrics = {
            "length": len(bert_title.split()),
            "character_count": len(bert_title)
        }

        return {
            "gemini_title": gemini_title,
            "bert_title": bert_title,
            "tuning_title": tuning_title,
            "gemini_metrics": gemini_metrics,
            "bert_metrics": bert_metrics,
            "tuning_metrics": tuning_metrics
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
