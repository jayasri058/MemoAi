import os

# Base Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
AUDIO_FOLDER = os.path.join(UPLOAD_FOLDER, 'audio')
IMAGE_FOLDER = os.path.join(UPLOAD_FOLDER, 'images')

# Persistence Files
MEMORY_FILE = os.path.join(BASE_DIR, 'memory.json')
INDEX_FILE = os.path.join(BASE_DIR, 'faiss.index')

# Make sure they exist
for folder in [UPLOAD_FOLDER, AUDIO_FOLDER, IMAGE_FOLDER]:
    os.makedirs(folder, exist_ok=True)
