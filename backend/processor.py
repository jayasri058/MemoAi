import os
import torch
import librosa
import numpy as np
import faiss
import uuid
import json
from datetime import datetime
from PIL import Image
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import google.generativeai as genai
from .config import MEMORY_FILE, INDEX_FILE

# Global state
transcriber = None
llm = None
embedder = None
index = None
memory = []

class Category(BaseModel):
    category: str = Field(description="The category of the input text")

def initialize_models():
    global transcriber, llm, embedder, index, memory
    
    print("Initializing models...")
    
    # Initialize Whisper
    device = 0 if torch.cuda.is_available() else -1
    try:
        transcriber = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-tiny.en",
            device=device
        )
        print("Whisper model loaded")
    except Exception as e:
        print(f"Whisper error: {e}")
        transcriber = None
    
    # Initialize LLM
    api_key = os.getenv("Gemini_api_key")
    if api_key:
        genai.configure(api_key=api_key)
        try:
            llm = GoogleGenerativeAI(api_key=api_key, model="gemini-2.0-flash")
            print("LLM loaded")
        except Exception as e:
            print(f"LLM error: {e}")
            llm = None
    
    # Initialize Embedding
    try:
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        print("Embedder loaded")
    except Exception as e:
        print(f"Embedder error: {e}")
        embedder = None
    
    # Load Memory & Index
    dimension = 384
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, 'r') as f:
                memory = json.load(f)
            print(f"Loaded {len(memory)} memories")
        except: memory = []
    else: memory = []
        
    if os.path.exists(INDEX_FILE):
        try:
            index = faiss.read_index(INDEX_FILE)
            print("Loaded FAISS index")
        except: index = faiss.IndexFlatL2(dimension)
    else:
        index = faiss.IndexFlatL2(dimension)
    
    print("Models initialized!")

def transcribe_audio(file_path):
    if not transcriber: return "Transcription service unavailable"
    try:
        audio, sr = librosa.load(file_path, sr=16000)
    except:
        # Fallback to direct path processing for Whisper
        result = transcriber(file_path)
        return result["text"]
    result = transcriber(audio)
    return result["text"]

def analyze_image(image_path):
    try:
        img = Image.open(image_path)
        model = genai.GenerativeModel('gemini-2.0-flash') # Using consistent flash model
        response = model.generate_content([
            "Describe this image clearly. Mention shop type, items, and visual context.",
            img
        ])
        return response.text
    except Exception as e:
        return f"Image analysis failed: {str(e)}"

def get_embedding(text):
    if not embedder: return np.random.rand(384).astype('float32')
    return embedder.encode(text)

def store_memory(embedding, record):
    global index, memory
    index.add(np.array([embedding]))
    memory.append(record)
    try:
        faiss.write_index(index, INDEX_FILE)
        with open(MEMORY_FILE, 'w') as f:
            json.dump(memory, f, indent=4)
    except Exception as e:
        print(f"Store error: {e}")

def search_memories(query_text, k=10):
    if not memory: return []
    query_text_lower = query_text.lower()
    query_words = set(query_text_lower.split())
    query_embedding = get_embedding(query_text)
    
    search_k = min(max(k * 3, 50), len(memory))
    distances, indices = index.search(np.array([query_embedding]), k=search_k)
    
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(memory): continue
        record = memory[idx].copy()
        semantic_score = 1.0 / (1.0 + dist)
        
        content_val = record.get('content') or ""
        context_val = record.get('context') or ""
        content_text = f"{content_val} {context_val}".lower()
        matched_words = query_words.intersection(set(content_text.split()))
        keyword_boost = 0.2 * len(matched_words)
        
        recency_boost = 0
        try:
            ts_val = record.get('timestamp')
            if ts_val:
                ts = datetime.fromisoformat(ts_val)
                hours_since = (datetime.now() - ts).total_seconds() / 3600
                if hours_since < 24:
                    recency_boost = 0.1 * (1.0 - (hours_since / 24.0))
        except: pass
        
        record['relevance_score'] = float(semantic_score + keyword_boost + recency_boost)
        results.append(record)
        
    results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
    return results

def categorize_text(text):
    if not llm: return "General"
    try:
        parser = PydanticOutputParser(pydantic_object=Category)
        prompt = PromptTemplate(
            input_variables=["input"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
            template="Categorize this text into one of: Daily Life, Work & Meetings, Learning & Growth, Health & Fitness, Money & Shopping, Entertainment & Leisure, Ideas & Creativity, General.\n{format_instructions}\nInput: {input}"
        )
        chain = prompt | llm | parser
        result = chain.invoke(text)
        return result.category
    except: return "General"
