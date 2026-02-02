"""
MemoAI Flask Backend Server
Handles API requests for memory processing, categorization, and search
"""

import os
import json
import base64
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash, check_password_hash
from models import get_db_manager

# PDF handling imports
try:
    from PyPDF2 import PdfReader
    PDF_SUPPORT = True
    print("PDF support enabled (PyPDF2)")
except ImportError:
    try:
        import pypdf
        from pypdf import PdfReader
        PDF_SUPPORT = True
        print("PDF support enabled (pypdf)")
    except ImportError:
        PDF_SUPPORT = False
        print("PDF support disabled. Install with: pip install PyPDF2")

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__, static_folder='.')
CORS(app)  # Enable CORS for all routes

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'pdf'}

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize database
db_manager = get_db_manager()

# Ensure database is properly initialized
try:
    # Test if FTS table exists
    db_manager.search_memories("test")
except Exception as e:
    print(f"Error during database search test: {e}")
    # Reinitialize database if needed
    db_manager.init_database()

# Initialize sentence transformer for embeddings
try:
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
except ImportError:
    print("Warning: sentence-transformers not installed. Install with: pip install sentence-transformers")
    embedder = None

# Initialize AI Services
from vector_store import PineconeManager
from ai_services import GeminiService

# Load API Keys
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
# Check both uppercase and capitalized versions
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY') or os.getenv('Gemini_api_key')

if not PINECONE_API_KEY:
    print("Warning: PINECONE_API_KEY not found in environment variables")
if not GEMINI_API_KEY:
    print("Warning: GEMINI_API_KEY not found in environment variables")

# Initialize Managers
pinecone_manager = PineconeManager(PINECONE_API_KEY)
gemini_service = GeminiService(GEMINI_API_KEY)
# Disable legacy FAISS
vector_index = None

# Load API keys
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    print("Warning: GEMINI_API_KEY not found in environment variables")

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_pdf_text(pdf_path):
    """Extract text from PDF file"""
    if not PDF_SUPPORT:
        return None
    
    try:
        reader = PdfReader(pdf_path)
        text_content = []
        
        for page_num, page in enumerate(reader.pages, 1):
            text = page.extract_text()
            if text.strip():
                text_content.append(f"[Page {page_num}]\n{text}")
        
        return "\n\n".join(text_content)
    except Exception as e:
        print(f"Error extracting PDF text: {e}")
        return None

def get_embedding(text):
    """Get embedding for text using sentence transformer"""
    if embedder and text:
        # Normalize the embedding for cosine similarity natively in the encoder
        embedding = embedder.encode([text], normalize_embeddings=True)
        return embedding[0]
    return np.zeros(384, dtype=np.float32)

def chunk_text(text, chunk_size=600, overlap=100):
    """Split text into chunks with overlap for better vector search"""
    chunks = []
    if not text:
        return chunks
        
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        if len(text) <= chunk_size:
            break
        start += (chunk_size - overlap)
        
    return chunks

def add_to_vector_index(memory_id, text, metadata=None):
    """Add a memory to the vector index for similarity search"""
    if pinecone_manager is None or not embedder:
        return
        
    embedding = get_embedding(text)
    
    # Enhance metadata
    if metadata is None:
        metadata = {}
    
    # Ensure ID info is part of metadata
    metadata['memory_id'] = memory_id
    metadata['text_preview'] = text[:500]
    metadata['created_at'] = int(datetime.now().timestamp())
    
    # Upsert to Pinecone
    pinecone_manager.upsert_vector(
        id=str(memory_id),
        vector=embedding.tolist(),
        metadata=metadata
    )

def add_pdf_to_vector_index(memory_id, pdf_text, metadata=None):
    """Split PDF into chunks and add multiple vectors to Pinecone"""
    if pinecone_manager is None or not embedder:
        return
        
    # Split text into chunks (600 chars is good for MiniLM)
    chunks = chunk_text(pdf_text, chunk_size=600, overlap=120)
    
    upsert_items = []
    import re
    
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        
        chunk_metadata = (metadata or {}).copy()
        chunk_metadata['text'] = chunk
        chunk_metadata['chunk_index'] = i
        chunk_metadata['memory_id'] = memory_id
        chunk_metadata['type'] = 'pdf_chunk'
        
        # Try to find page number if present in chunk
        page_match = re.search(r'\[Page (\d+)\]', chunk)
        if page_match:
            chunk_metadata['page_number'] = int(page_match.group(1))
            
        # ID: memoryID_chunkIndex
        upsert_items.append((f"{memory_id}_{i}", embedding.tolist(), chunk_metadata))
    
    # Batch upsert to Pinecone
    pinecone_manager.upsert_vectors(upsert_items)

def search_similar_vectors(query_text, top_k=5, threshold=0.3):
    """Search for similar vectors using Pinecone"""
    if pinecone_manager is None or not embedder:
        return []
    
    embedding = get_embedding(query_text)
    
    # Search with threshold to fix "irrelevant results" issue
    results = pinecone_manager.query_similarity(
        vector=embedding.tolist(),
        top_k=top_k,
        threshold=threshold
    )
    
    # Map results
    mapped_results = []
    seen_memories = set()
    
    for res in results:
        try:
            # Handle chunked IDs (e.g., "123_0")
            raw_id = res['id']
            if '_' in raw_id:
                mem_id = int(raw_id.split('_')[0])
            else:
                mem_id = int(raw_id)
                
            # If we've already seen this memory (matched another chunk), skip unless we want multi-snippet support
            # For simplicity, we take the highest scoring chunk for each memory
            if mem_id in seen_memories:
                continue
            seen_memories.add(mem_id)
            
            mapped_results.append({
                'memory_id': mem_id,
                'similarity_score': float(res['score']),
                'metadata': res.get('metadata', {})
            })
        except ValueError:
            continue
            
    return mapped_results

@app.route('/')
def serve_index():
    """Serve the main HTML file"""
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    """Serve static files"""
    try:
        return send_from_directory('.', path)
    except FileNotFoundError:
        return jsonify({'error': 'File not found'}), 404

@app.route('/api/process-memory', methods=['POST'])
def process_memory():
    """
    Process memory input (voice text + optional image)
    Returns categorized memory with context and tags
    """
    try:
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        voice_text = data.get('voice_text', '').strip()
        has_image = data.get('has_image', False)
        
        # ... rest of the function ...


        image_data = data.get('image_data', None)
        
        # Validate input
        if not voice_text:
            return jsonify({'error': 'Voice text is required'}), 400
        
        # Save image if provided
        image_path = None
        if has_image and image_data:
            try:
                # Decode base64 image data
                header, encoded = image_data.split(',', 1)
                image_bytes = base64.b64decode(encoded)
                
                # Generate unique filename
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'memory_{timestamp}.png'
                image_path = os.path.join(UPLOAD_FOLDER, filename)
                
                # Save image file
                with open(image_path, 'wb') as f:
                    f.write(image_bytes)
                    
            except Exception as e:
                print(f"Error saving image: {e}")
                # Continue processing without image
        
        # Process the memory
        result = process_memory_logic(voice_text, image_path)
        
        # Save to database
        memory_id = db_manager.save_memory({
            'title': result.get('title', ''),
            'content': result.get('content', ''),
            'voice_text': voice_text,
            'category': result.get('category', ''),
            'context': result.get('context', ''),
            'tags': result.get('tags', []),
            'image_path': image_path
        })
        
        # Determine if it's a PDF for specialized indexing
        is_pdf = image_path and image_path.lower().endswith('.pdf')
        
        # Add to vector index for similarity search
        search_text = f"{result.get('content', '')} {result.get('context', '')} {' '.join(result.get('tags', []))}"
        
        # Metadata for Pinecone filtering
        vector_metadata = {
            'category': result.get('category', 'General'),
            'tags': result.get('tags', []),
            'has_image': bool(image_path) and not is_pdf,
            'date': datetime.now().strftime('%Y-%m-%d'),
            'type': 'pdf' if is_pdf else ('image' if image_path else 'voice')
        }
        
        if is_pdf:
            add_pdf_to_vector_index(memory_id, result.get('content', ''), vector_metadata)
        else:
            add_to_vector_index(memory_id, search_text, vector_metadata)
        
        # Add metadata
        result['id'] = memory_id
        result['timestamp'] = datetime.now().isoformat()
        result['processed_successfully'] = True
        
        return jsonify(result), 200
        
    except Exception as e:
        print(f"Error processing memory: {e}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@app.route('/api/register', methods=['POST'])
def register():
    """Register a new user"""
    try:
        data = request.get_json()
        name = data.get('name')
        email = data.get('email')
        password = data.get('password')
        
        if not all([name, email, password]):
            return jsonify({'error': 'Missing required fields'}), 400
            
        # Check if user already exists
        if db_manager.get_user_by_email(email):
            return jsonify({'error': 'Email already registered'}), 409
            
        # hash password
        password_hash = generate_password_hash(password)
        
        # Create user
        if db_manager.create_user(name, email, password_hash):
            return jsonify({'message': 'User registered successfully'}), 201
        else:
            return jsonify({'error': 'Failed to register user'}), 500
            
    except Exception as e:
        print(f"Error registering user: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/login', methods=['POST'])
def login():
    """Login user"""
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        
        if not all([email, password]):
            return jsonify({'error': 'Missing credentials'}), 400
            
        user = db_manager.get_user_by_email(email)
        
        if user and check_password_hash(user['password_hash'], password):
            return jsonify({
                'message': 'Login successful',
                'user': {
                    'id': user['id'],
                    'name': user['name'],
                    'email': user['email']
                }
            }), 200
        else:
            return jsonify({'error': 'Invalid email or password'}), 401
            
    except Exception as e:
        print(f"Error logging in: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/auth/google/accounts', methods=['GET'])
def get_google_accounts():
    """Get all registered Google accounts for account selector"""
    try:
        # In a real app, this would fetch accounts based on OAuth provider
        # For demo purposes, we'll return all registered users
        users = db_manager.get_all_users()
        
        if not users:
            # If no users exist, create some mock Google accounts
            mock_accounts = [
                {"name": "Jayasri", "email": "jayasri058@gmail.com"},
                {"name": "John Doe", "email": "john.doe@gmail.com"},
                {"name": "Jane Smith", "email": "jane.smith@gmail.com"}
            ]
            
            dummy_password = "google_oauth_secure_password"
            password_hash = generate_password_hash(dummy_password)
            
            for account in mock_accounts:
                if not db_manager.get_user_by_email(account['email']):
                    db_manager.create_user(account['name'], account['email'], password_hash)
            
            # Fetch again after creating mock accounts
            users = db_manager.get_all_users()
        
        # Format user data for frontend
        accounts = [
            {
                'id': user['id'],
                'name': user['name'],
                'email': user['email']
            }
            for user in users
        ]
        
        return jsonify({
            'accounts': accounts,
            'count': len(accounts)
        }), 200
        
    except Exception as e:
        print(f"Error fetching Google accounts: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/auth/google', methods=['POST'])
def google_auth():
    """Authenticate with selected Google account"""
    try:
        data = request.get_json()
        email = data.get('email')
        
        if not email:
            return jsonify({'error': 'Email is required'}), 400
        
        # Check if user exists
        user = db_manager.get_user_by_email(email)
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
            
        return jsonify({
            'message': 'Google login successful',
            'user': {
                'id': user['id'],
                'name': user['name'],
                'email': user['email']
            }
        }), 200
            
    except Exception as e:
        print(f"Error in Google auth: {e}")
        return jsonify({'error': 'Internal server error'}), 500

def process_memory_logic(voice_text, image_path=None):
    """
    Core memory processing logic using Gemini
    """
    # Use Gemini for analyzing image if present
    gemini_description = ""
    if image_path and gemini_service:
        print(f"Analyzing image with Gemini: {image_path}")
        gemini_description = gemini_service.describe_image(image_path)
    
    # Combined text for analysis
    full_text = f"{voice_text}\n{gemini_description}".strip()
    
    # Use Gemini for tagging (if available) or fallback
    tags = []
    if gemini_service and full_text:
        tags = gemini_service.generate_tags(full_text)
    
    if not tags:
        tags = generate_tags(full_text) # Fallback logic
        
    # Categorize (using existing logic for speed, or could upgrade to Gemini)
    category = classify_category(full_text)
    
    # Generate context
    context = ""
    if gemini_description:
        context = f"Image Analysis: {gemini_description}"
    else:
        context = generate_context(voice_text, None)
    
    return {
        'category': category,
        'context': context,
        'tags': tags,
        'title': generate_title(full_text),
        'content': full_text,
        'image_description': gemini_description,
        'voice_text': voice_text,
        'image_path': image_path
    }

def classify_category(text):
    """Simulate category classification"""
    text_lower = text.lower()
    
    categories = {
        'Daily Life': ['home', 'family', 'personal', 'daily', 'today', 'morning', 'evening'],
        'Work & Meetings': ['work', 'meeting', 'office', 'colleagues', 'project', 'presentation'],
        'Learning & Growth': ['learn', 'study', 'education', 'growth', 'improve', 'knowledge'],
        'Health & Fitness': ['health', 'exercise', 'fitness', 'diet', 'wellness', 'medical'],
        'Money & Shopping': ['money', 'buy', 'purchase', 'shop', 'price', 'budget', 'finance'],
        'Entertainment & Leisure': ['movie', 'music', 'game', 'fun', 'relax', 'entertainment'],
        'Ideas & Creativity': ['idea', 'creative', 'innovation', 'design', 'think', 'brainstorm']
    }
    
    for category, keywords in categories.items():
        if any(keyword in text_lower for keyword in keywords):
            return category
    
    return 'General'

def generate_context(voice_text, image_path):
    """Generate context description"""
    if not voice_text and not image_path:
        return 'No content to analyze.'
    
    context = ''
    
    if voice_text:
        context += f'Voice content: "{voice_text}". '
    
    if image_path:
        context += 'Associated with an uploaded image. '
    
    if voice_text and image_path:
        context += 'Combining auditory and visual information for enhanced context understanding.'
    elif voice_text:
        context += 'Audio-based memory with textual context.'
    elif image_path:
        context += 'Visual memory with potential for detailed analysis.'
    
    return context

def generate_tags(text):
    """Generate relevant tags"""
    if not text:
        return ['general']
    
    text_lower = text.lower()
    tags = []
    
    # Extract potential tags based on common patterns
    tag_keywords = {
        'meeting': ['meeting'],
        'project': ['project'],
        'idea': ['idea'],
        'health': ['health'],
        'fitness': ['exercise'],
        'shopping': ['shopping'],
        'learning': ['learn', 'study'],
        'work': ['work'],
        'family': ['family'],
        'social': ['friend'],
        'travel': ['travel'],
        'food': ['food']
    }
    
    for tag, keywords in tag_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            tags.append(tag)
    
    # Add time-related tags
    if any(word in text_lower for word in ['today', 'now']):
        tags.append('today')
    if 'tomorrow' in text_lower:
        tags.append('tomorrow')
    if 'yesterday' in text_lower:
        tags.append('past')
    
    # If no specific tags found, use general
    if not tags:
        tags.append('general')
    
    return list(set(tags))  # Remove duplicates

def generate_title(text):
    """Generate a title from the text"""
    if not text:
        return 'Untitled Memory'
    
    # Take first few words as title
    words = text.split()[:5]
    title = ' '.join(words)
    
    # Add ellipsis if truncated
    if len(text.split()) > 5:
        title += '...'
    
    return title

@app.route('/api/search-memories', methods=['GET'])
def search_memories():
    """
    Search through stored memories using vector similarity
    """
    try:
        query = request.args.get('q', '').strip().lower()
        
        if not query:
            return jsonify({'error': 'Search query is required'}), 400
        
        # First, try vector similarity search if available
        similar_results = []
        if pinecone_manager is not None and embedder:
            # Similarity threshold 0.3 filters out irrelevant results
            similar_results = search_similar_vectors(query, top_k=10, threshold=0.3)
        
        # Get memory details for similar results
        similar_results_memories = []
        for res in similar_results:
            memory_id = res['memory_id']
            memory = db_manager.get_memory(memory_id)
            if memory:
                memory['similarity_score'] = res['similarity_score']
                
                # Extract image description if not present but in context
                metadata = res.get('metadata', {})
                memory['image_description'] = metadata.get('image_description', '')
                
                if not memory.get('image_description') and memory.get('context', '').startswith('Image Analysis: '):
                    memory['image_description'] = memory['context'].replace('Image Analysis: ', '')
                
                # If it's a PDF chunk, include specific metadata for UI
                metadata = res.get('metadata', {})
                if metadata.get('type') == 'pdf_chunk':
                    memory['is_pdf_chunk'] = True
                    memory['page_number'] = metadata.get('page_number')
                    # Override snippet to show the specific relevant chunk
                    chunk_text = metadata.get('text', '')
                    page_info = f"[Page {memory['page_number']}] " if memory['page_number'] else ""
                    memory['snippet'] = f"{page_info}{chunk_text[:200]}..."
                
                similar_results_memories.append(memory)
        
        detailed_results = similar_results_memories
        
        # If vector search didn't return enough results, fall back to text search
        if len(detailed_results) < 5:
            text_results = db_manager.search_memories(query)
            # Combine and deduplicate results
            existing_ids = {mem['id'] for mem in detailed_results}
            for text_result in text_results:
                if text_result['id'] not in existing_ids:
                    detailed_results.append(text_result)
                    existing_ids.add(text_result['id'])
        
        # Sort by similarity score if available, otherwise by date
        detailed_results.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
        
        return jsonify({
            'results': detailed_results,
            'count': len(detailed_results),
            'query': query,
            'search_type': 'vector_similarity' if similar_results else 'text_fallback'
        }), 200
        
    except Exception as e:
        print(f"Error searching memories: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0',
        'vector_search_enabled': vector_index is not None
    }), 200

if __name__ == '__main__':
    print("Starting MemoAI Backend Server...")
    # Check both legacy and new pinecone manager
    is_vector_enabled = (vector_index is not None) or (pinecone_manager and pinecone_manager.index is not None)
    print(f"Vector search enabled: {'Yes' if is_vector_enabled else 'No'}")
    print(f"Gemini API Key configured: {'Yes' if GEMINI_API_KEY else 'No'}")
    print("Server running on http://localhost:5000")
    
    # Run the Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )