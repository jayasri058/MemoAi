# MemoAI — Technical Documentation

> **Version:** 2.0.0  
> **Last Updated:** February 25, 2026  
> **Python:** ≥ 3.10 | **Framework:** Flask | **Database:** Pinecone + SQLite

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Project Structure](#3-project-structure)
4. [Technology Stack](#4-technology-stack)
5. [Core Modules](#5-core-modules)
   - 5.1 [app.py — Flask Backend Server](#51-apppy--flask-backend-server)
   - 5.2 [models.py — Database Manager](#52-modelspy--database-manager)
   - 5.3 [vector_store.py — Pinecone Storage Layer](#53-vector_storepy--pinecone-storage-layer)
   - 5.4 [ai_services.py — AI / ML Services](#54-ai_servicespy--ai--ml-services)
   - 5.5 [backend/processor.py — Legacy Processing Engine](#55-backendprocessorpy--legacy-processing-engine)
6. [API Reference](#6-api-reference)
7. [Data Models & Schema](#7-data-models--schema)
8. [Authentication & Authorization](#8-authentication--authorization)
9. [AI / ML Pipeline](#9-ai--ml-pipeline)
10. [Vector Search Architecture](#10-vector-search-architecture)
11. [Frontend Architecture](#11-frontend-architecture)
12. [Configuration & Environment](#12-configuration--environment)
13. [Deployment](#13-deployment)
14. [Security Considerations](#14-security-considerations)
15. [Rate Limiting & Monetization](#15-rate-limiting--monetization)
16. [Error Handling & Logging](#16-error-handling--logging)
17. [Known Limitations & Future Work](#17-known-limitations--future-work)

---

## 1. Project Overview

**MemoAI** is an intelligent personal memory assistant that helps users capture, categorize, and retrieve their thoughts, conversations, and visual memories. It combines voice recognition, image analysis, AI-powered categorization, and semantic vector search to act as a user's "second brain."

### Key Capabilities

| Capability | Description |
|---|---|
| **Voice Capture** | Browser-based speech recognition converts spoken thoughts to text |
| **Image Analysis** | Google Gemini Vision analyzes uploaded images; BLIP model as fallback |
| **Smart Categorization** | AI-powered + keyword-based classification into 8 categories |
| **Semantic Search** | Vector similarity search via Pinecone + text-based fallback |
| **AI Summaries** | Gemini-powered summaries of a user's recent memory activity |
| **User Accounts** | Registration, login, Google-style account selection |
| **Freemium Model** | 10 free memories; premium upgrade via payment gateway |

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLIENT (Browser)                        │
│  ┌──────────┐  ┌───────────┐  ┌───────────────┐  │
│  │ Voice    │  │ Image     │  │ Search &      │  │
│  │ Recorder │  │ Upload    │  │ Dashboard     │  │
│  └────┬─────┘  └─────┬─────┘  └──────┬────────┘  │
│       └──────────────┼───────────────┘            │
│                      ▼                                          │
│              REST API (JSON / Base64)                            │
└──────────────────────┬──────────────────────────────────────────┘
                       │ HTTP
┌──────────────────────▼──────────────────────────────────────────┐
│                    FLASK SERVER (app.py)                         │
│  ┌─────────────┐ ┌──────────────┐ ┌───────────────────────────┐ │
│  │ Auth Routes │ │ Memory CRUD  │ │ Search / Summary Routes   │ │
│  │ /api/login  │ │ /api/process │ │ /api/search-memories      │ │
│  │ /api/register│ │ -memory     │ │ /api/memories/summary     │ │
│  └──────┬──────┘ └──────┬───────┘ └────────────┬──────────────┘ │
│         │               │                      │                │
│  ┌──────▼────┐   ┌──────▼──────┐   ┌───────────▼────────────┐  │
│  │ models.py │   │ai_services  │   │ Sentence Transformer   │  │
│  │ (DB Mgr)  │   │(Gemini/BLIP)│   │ (all-MiniLM-L6-v2)    │  │
│  └──┬────┬───┘   └─────────────┘   └────────────────────────┘  │
│     │    │                                                      │
└─────┼────┼──────────────────────────────────────────────────────┘
      │    │
 ┌────▼──┐ ┌▼────────────────┐
 │SQLite │ │   Pinecone DB   │
 │Users  │ │ (Vectors +      │
 │Auth   │ │  Memory Data)   │
 └───────┘ └─────────────────┘
```

### Data Flow — Memory Creation

```
User speaks / uploads image
       │
       ▼
Browser captures text/image (SpeechRecognition API / FileReader)
       │
       ▼
POST /api/process-memory  {voice_text, image_data}
       │
       ▼
┌─ Memory Limit Check (SQLite: memory_count, is_premium)
│      │
│      ▼
│  Image Processing (save to /uploads, strip EXIF metadata)
│      │
│      ▼
│  AI Analysis
│  ├─ Gemini Vision → image description
│  ├─ Gemini Text  → tag generation
│  └─ Keyword engine → category, context, title, fallback tags
│      │
│      ▼
│  Embedding Generation (SentenceTransformer: all-MiniLM-L6-v2)
│      │
│      ▼
│  Save to Pinecone (vector + metadata in "memories" namespace)
│      │
│      ▼
│  Increment memory_count in SQLite
│      │
│      ▼
└─ Return JSON response with memory ID, category, tags, usage info
```

---

## 3. Project Structure

```
memo-ai/
│
├── app.py                  # Main Flask server — all API routes & orchestration
├── models.py               # DatabaseManager — bridges SQLite auth + Pinecone memory
├── vector_store.py          # PineconeManager — all Pinecone CRUD operations
├── ai_services.py           # GeminiService — image description, tag generation
│
├── backend/
│   ├── __init__.py          # Package init
│   ├── config.py            # Path constants (BASE_DIR, UPLOAD_FOLDER, etc.)
│   └── processor.py         # Legacy processing engine (Whisper, FAISS, LangChain)
│
├── templates/
│   └── index.html           # Landing page / demo page (Jinja2 template)
│
├── static/
│   ├── css/
│   │   └── styles.css       # Landing page styles
│   └── js/
│       └── script.js        # Landing page JavaScript
│
├── index.html               # Main app dashboard (served as static file)
├── login.html               # Login page
├── register.html            # Registration page
├── contact.html             # Contact page
├── script.js                # Main app JavaScript (1630 lines)
├── styles.css               # Main app CSS (47,679 bytes)
│
├── requirements.txt         # pip dependencies
├── pyproject.toml           # Project metadata & uv/pip dependencies
├── Procfile                 # Deployment process file
├── .env.example             # Environment variable template
├── .gitignore               # Git ignore rules
├── .python-version          # Python version lock
└── uv.lock                  # uv package manager lockfile
```

### Auxiliary / Utility Files

| File | Purpose |
|---|---|
| `explore_vectors.py` | Debug utility to inspect Pinecone vector data |
| `inspect_db.py` | Debug utility to inspect SQLite database |
| `migrate_to_vector.py` | Migration script from legacy storage to Pinecone |
| `min_pinecone_test.py` | Minimal Pinecone connection test |
| `test_services.py` | Test script for AI services |
| `test_image.py` | Test script for image processing |
| `debug_output.txt` | Debug output log |
| `memory.json` | Legacy memory store (JSON file) |
| `faiss.index` / `vector_index.faiss` | Legacy FAISS vector indexes |
| `vector_memory.pkl` | Legacy pickled vector memory |
| `memoai.db` | SQLite database for user authentication |

---

## 4. Technology Stack

### Backend

| Technology | Version | Purpose |
|---|---|---|
| **Python** | ≥ 3.10 | Runtime |
| **Flask** | ≥ 3.1.2 | Web framework |
| **Flask-CORS** | ≥ 6.0.2 | Cross-origin resource sharing |
| **Flask-Limiter** | — | Rate limiting |
| **Werkzeug** | — | Password hashing (`generate_password_hash`) |
| **python-dotenv** | ≥ 1.2.1 | Environment variable management |
| **SQLite** | Built-in | User authentication storage |

### AI / ML

| Technology | Version | Purpose |
|---|---|---|
| **Google Generative AI (Gemini)** | ≥ 0.8.6 | Image analysis, tag generation, summaries |
| **Sentence Transformers** | ≥ 5.2.0 | Text embedding generation (`all-MiniLM-L6-v2`) |
| **Transformers (HuggingFace)** | ≥ 4.57.3 | BLIP fallback for image captioning |
| **PyTorch** | ≥ 2.9.1 | ML framework backend |
| **LangChain** | ≥ 1.2.3 | LLM orchestration (used in legacy processor) |
| **Pillow** | ≥ 12.1.0 | Image processing, EXIF metadata stripping |

### Database & Vector Storage

| Technology | Purpose |
|---|---|
| **Pinecone** (Serverless, AWS us-east-1) | Primary vector database — stores memories + user accounts |
| **SQLite** (`memoai.db`) | User authentication (register, login, memory count, premium status) |

### Frontend

| Technology | Purpose |
|---|---|
| **HTML5** | Page structure |
| **CSS3** | Styling, animations, dark/light theme |
| **Vanilla JavaScript** | Client-side logic, speech recognition, API calls |
| **Web Speech API** | Browser-native speech recognition |
| **Font Awesome** | Icon library |
| **Google Fonts (Poppins)** | Typography |

---

## 5. Core Modules

### 5.1 `app.py` — Flask Backend Server

**Lines:** 931 | **Role:** Central orchestration layer

This is the main entry point of the application. It initializes all services, defines all API routes, and contains the core memory processing logic.

#### Key Responsibilities

- **Flask App Initialization:** Configures CORS, rate limiting, file upload folder
- **Service Initialization:** Loads Gemini API, Sentence Transformer, Pinecone manager
- **Route Definitions:** All REST API endpoints (auth, memory CRUD, search, payment)
- **Memory Processing Pipeline:** `process_memory_logic()` orchestrates AI analysis
- **Vector Operations:** Embedding generation, similarity search

#### Important Functions

| Function | Lines | Description |
|---|---|---|
| `get_embedding(text)` | 137–142 | Generates 384-dim embedding via SentenceTransformer |
| `add_to_vector_index(memory_id, text, user_id, metadata)` | 161–180 | Upserts a single memory vector to Pinecone |
| `search_similar_vectors(query_text, user_id, top_k, threshold)` | 212–249 | Queries Pinecone for semantically similar memories |
| `process_memory_logic(voice_text, image_path)` | 632–667 | Core AI pipeline: image analysis → tags → category → context |
| `classify_category(text)` | 669–687 | Keyword-based category classification (8 categories) |
| `generate_tags(text)` | 711–748 | Rule-based tag generation fallback |
| `generate_title(text)` | 750–761 | Generates title from first 5 words of text |
| `login_required(f)` | 99–111 | Decorator for authenticated endpoints |

#### Initialization Sequence

```python
1. load_dotenv()                           # Load .env
2. Flask(__name__, static_folder='.')       # Create app with current dir as static root
3. CORS(app)                               # Enable CORS
4. Limiter(key_func=get_user_id_as_key)    # Rate limiting by user ID or IP
5. get_db_manager()                        # Initialize DatabaseManager → PineconeManager
6. SentenceTransformer('all-MiniLM-L6-v2') # Load embedding model
7. GeminiService(GEMINI_API_KEY)           # Initialize Gemini AI
```

---

### 5.2 `models.py` — Database Manager

**Lines:** 304 | **Role:** Unified data access layer

The `DatabaseManager` class provides a unified interface across two storage backends:

| Operation | Backend | Rationale |
|---|---|---|
| User registration / login | **SQLite** (`memoai.db`) | Reliable, always works locally, no network dependency |
| Memory CRUD | **Pinecone** | Vector-native storage, enables semantic search |

#### Class: `DatabaseManager`

```python
class DatabaseManager:
    def __init__(self):
        # Initializes PineconeManager with API key and index name
        self.pinecone = PineconeManager(api_key, index_name)
```

#### User Operations (SQLite)

| Method | Signature | Description |
|---|---|---|
| `create_user` | `(name, email, password_hash) → bool` | Registers user; returns `False` if email exists |
| `get_user_by_email` | `(email) → Optional[Dict]` | Lookup by email |
| `get_user_by_id` | `(user_id) → Optional[Dict]` | Lookup by ID (includes premium status) |
| `get_all_users` | `() → List[Dict]` | Returns all registered users |
| `get_memory_count` | `(user_id) → Tuple[int, bool]` | Returns `(count, is_premium)` |
| `increment_memory_count` | `(user_id) → int` | Increments and returns new count |
| `set_premium` | `(user_id) → bool` | Marks user as premium |

#### Memory Operations (Delegated to Pinecone)

| Method | Signature | Description |
|---|---|---|
| `save_memory` | `(user_id, memory_data, vector) → int` | Saves memory with embedding |
| `get_memory` | `(memory_id, user_id) → Optional[Dict]` | Fetches single memory by ID |
| `get_all_memories` | `(user_id) → List[Dict]` | All user memories, resolves legacy IDs |
| `search_memories` | `(query, user_id) → List[Dict]` | Text-based search fallback |
| `delete_memory` | `(memory_id, user_id) → bool` | Deletes memory |
| `update_memory` | `(memory_id, update_data, user_id) → bool` | Updates memory metadata |

#### Legacy ID Resolution

The `_resolve_user_ids()` method handles migration from the old Pinecone-only auth system. It maps a current SQLite user ID to any legacy Pinecone user ID so that users can still access memories stored under the old system.

```python
def _resolve_user_ids(self, user_id: int) -> List[int]:
    ids = [user_id]
    # Looks up user email → checks Pinecone for legacy account
    # Appends legacy ID if found and different from current ID
    return ids
```

#### SQLite Schema

```sql
CREATE TABLE users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT NOT NULL,
    email         TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    memory_count  INTEGER DEFAULT 0,
    is_premium    INTEGER DEFAULT 0,
    created_at    TEXT DEFAULT CURRENT_TIMESTAMP
);
```

The table is auto-created at module import time via `_ensure_users_table()`, with automatic migration for `memory_count` and `is_premium` columns.

---

### 5.3 `vector_store.py` — Pinecone Storage Layer

**Lines:** 544 | **Role:** Unified Pinecone interface

The `PineconeManager` class handles all Pinecone interactions, using **namespaces** to separate data types:

| Namespace | Contents |
|---|---|
| `users` | User account records (legacy; auth now primarily in SQLite) |
| `memories` | Memory vectors + full metadata |

#### Configuration

| Parameter | Default | Description |
|---|---|---|
| `api_key` | From `PINECONE_API_KEY` env var | Pinecone authentication |
| `index_name` | `memo-ai-index` | Pinecone index name |
| `dimension` | `384` | Vector dimensionality (matches `all-MiniLM-L6-v2`) |
| `metric` | `cosine` | Similarity metric |
| `cloud` | `aws` | Serverless cloud provider |
| `region` | `us-east-1` | Serverless region |

#### Index Auto-Creation

On initialization, the manager checks if the Pinecone index exists. If not, it creates a new serverless index and waits for it to become ready:

```python
self.pc.create_index(
    name=index_name,
    dimension=384,
    metric='cosine',
    spec=ServerlessSpec(cloud='aws', region='us-east-1')
)
```

#### Memory Metadata Schema

When a memory is saved to Pinecone, the following metadata is stored alongside the vector:

```python
metadata = {
    "memory_id":   int,           # Unique memory identifier
    "user_id":     int,           # Owner user ID
    "title":       str,           # Generated title
    "content":     str (≤1000),   # Full text content (truncated for metadata limits)
    "voice_text":  str (≤1000),   # Original voice input
    "category":    str,           # AI-classified category
    "context":     str (≤1000),   # Generated context description
    "tags":        str (JSON),    # JSON-serialized list of tags
    "image_path":  str,           # Path to associated image file
    "created_at":  str (ISO),     # Creation timestamp
    "updated_at":  str (ISO),     # Last update timestamp
    "type":        str,           # "memory" | "image" | "voice"
    "has_image":   bool,          # Whether memory has an image
    "date":        str (YYYY-MM-DD) # Date for filtering
}
```

#### Key Operations

| Method | Description |
|---|---|
| `save_memory(user_id, memory_data, vector)` | Upserts a single vector with metadata |

---

### 5.4 `ai_services.py` — AI / ML Services

**Lines:** 96 | **Role:** AI-powered content analysis

#### Class: `GeminiService`

| Property | Description |
|---|---|
| `model` | `GenerativeModel('gemini-2.5-flash')` — primary AI model |
| `blip_processor` | BLIP image captioning processor (HuggingFace fallback) |
| `blip_model` | BLIP image captioning model (HuggingFace fallback) |

#### Methods

| Method | Input | Output | Description |
|---|---|---|---|
| `describe_image(image_path)` | File path | `str` | Uploads image to Gemini, generates detailed description. Falls back to `fallback_describe_image()` on failure. |
| `fallback_describe_image(image_path)` | File path | `str` | Uses local BLIP model (`Salesforce/blip-image-captioning-base`) for offline image captioning. |
| `generate_tags(text)` | Text | `list[str]` | Asks Gemini to analyze text and return 3–5 lowercase tags. |

#### Fallback Chain

```
Image Description:
  1. Gemini Vision (gemini-2.5-flash) — Primary
  2. BLIP (Salesforce/blip-image-captioning-base) — Local fallback
  3. Error message — Ultimate fallback

Tag Generation:
  1. Gemini text analysis — Primary
  2. Keyword-based regex engine (in app.py) — Fallback
```

---

### 5.5 `backend/processor.py` — Legacy Processing Engine

**Lines:** 174 | **Role:** Original processing pipeline (partially superseded)

This module contains the initial implementation that used FAISS for vector storage and Whisper for speech-to-text. It is retained for potential offline/local processing but the main app now uses Pinecone + browser Speech API.

#### Key Components

| Component | Technology | Status |
|---|---|---|
| Speech-to-Text | OpenAI Whisper (`whisper-tiny.en`) | Available (fallback) |
| Categorization | LangChain + Gemini (`gemini-2.0-flash`) | Available |
| Embeddings | SentenceTransformer (`all-MiniLM-L6-v2`) | Active (shared model) |
| Vector Store | FAISS (`IndexFlatL2`) | Legacy (replaced by Pinecone) |
| Memory Store | JSON file (`memory.json`) | Legacy |
| Image Analysis | Gemini Vision (`gemini-2.0-flash`) | Available |

#### Search Scoring (Legacy)

The legacy `search_memories()` function uses a composite scoring system:

```
relevance_score = semantic_score + keyword_boost + recency_boost
```

| Factor | Formula | Weight |
|---|---|---|
| Semantic Score | `1.0 / (1.0 + FAISS_distance)` | Primary |
| Keyword Boost | `0.2 × matched_word_count` | +0.2 per match |
| Recency Boost | `0.1 × (1.0 - hours_since/24)` | Max +0.1 (within 24h) |

---

## 6. API Reference

All API endpoints are served from the Flask app at `http://localhost:5000`.

### Authentication Endpoints

#### `POST /api/register`

Register a new user account.

**Request Body:**
```json
{
  "name": "Jane Doe",
  "email": "jane@example.com",
  "password": "securepassword"
}
```

**Responses:**
| Status | Description |
|---|---|
| `201` | User registered successfully |
| `400` | Missing required fields |
| `409` | Email already registered |
| `500` | Internal server error |

---

#### `POST /api/login`

Authenticate a user with email and password.

**Request Body:**
```json
{
  "email": "jane@example.com",
  "password": "securepassword"
}
```

**Success Response (200):**
```json
{
  "message": "Login successful",
  "user": {
    "id": 1,
    "name": "Jane Doe",
    "email": "jane@example.com"
  }
}
```

**Error Responses:**
| Status | Description |
|---|---|
| `400` | Missing credentials |
| `401` | Invalid email or password |

---

#### `GET /api/auth/google/accounts`

Get all registered accounts for the Google-style account selector.

**Response (200):**
```json
{
  "accounts": [
    {"id": 1, "name": "Jane Doe", "email": "jane@example.com"}
  ],
  "count": 1
}
```

---

#### `POST /api/auth/google`

Authenticate or auto-register via Google account selection.

**Request Body:**
```json
{ "email": "jane@example.com" }
```

**Response (200):**
```json
{
  "message": "Google login successful",
  "user": { "id": 1, "name": "Jane Doe", "email": "jane@example.com" }
}
```

---

### Memory Endpoints

> **All memory endpoints require the `X-User-Id` header.**

#### `POST /api/process-memory` 🔒

Process and store a new memory (voice text + optional image).

**Headers:**
```
X-User-Id: 1
```

**Request Body:**
```json
{
  "voice_text": "I had a great meeting today about the new project",
  "has_image": true,
  "image_data": "data:image/png;base64,iVBOR..."
}
```

**Success Response (200):**
```json
{
  "id": 1234567890,
  "title": "I had a great...",
  "content": "I had a great meeting today about the new project",
  "category": "Work & Meetings",
  "context": "Voice content: \"I had a great meeting today...\"",
  "tags": ["meeting", "work", "project"],
  "image_description": "",
  "timestamp": "2026-02-25T18:30:00.000000",
  "processed_successfully": true,
  "memories_used": 3,
  "memory_limit": 10,
  "is_premium": false
}
```

**Error Responses:**
| Status | Description |
|---|---|
| `400` | No data or missing voice_text |
| `401` | Authentication required |
| `402` | Memory limit reached (payment required) |
| `500` | Internal server error |

---

#### `GET /api/search-memories?q={query}` 🔒

Search stored memories using semantic vector search + text fallback.

**Query Parameters:**
| Parameter | Required | Description |
|---|---|---|
| `q` | Yes | Search query (use `*` or empty for all memories) |

**Rate Limit:** 50 per day

**Success Response (200):**
```json
{
  "results": [
    {
      "id": 1234567890,
      "title": "Meeting notes",
      "content": "...",
      "category": "Work & Meetings",
      "tags": ["meeting"],
      "similarity_score": 0.87,
      "created_at": "2026-02-25T18:30:00"
    }
  ],
  "count": 5,
  "query": "meeting notes",
  "search_type": "pinecone_vector"
}
```

**Search Strategy:**
1. If query is `*` or empty → return all memories for the user
2. Generate embedding for query text
3. Query Pinecone for top-10 similar vectors (threshold ≥ 0.3)
4. If < 5 results → supplement with text-based search fallback
5. Sort by similarity score descending
6. Return merged, deduplicated results

---

#### `GET /api/memories/summary` 🔒

Generate an AI summary of the user's recent memories.

**Rate Limit:** 10 per day

**Response (200):**
```json
{
  "summary": "You've been focused on work projects and health goals this week...",
  "count": 15
}
```

---

### Usage & Payment Endpoints

#### `GET /api/user/usage` 🔒

Get memory usage stats for the authenticated user.

**Response (200):**
```json
{
  "memories_used": 7,
  "memory_limit": 10,
  "remaining": 3,
  "is_premium": false,
  "limit_reached": false
}
```

---

#### `POST /api/payment/initiate` 🔒

Create a payment order for premium upgrade.

**Response (200):**
```json
{
  "order_id": "order_A1B2C3D4E5F6G7H8",
  "amount": 299,
  "currency": "INR",
  "price_display": "₹299/month",
  "description": "MemoAI Premium — Unlimited Memories"
}
```

> **Note:** Currently returns a simulated order. Razorpay/Stripe integration code is commented out but ready for activation.

---

#### `POST /api/payment/verify` 🔒

Verify payment and upgrade user to premium.

**Request Body:**
```json
{
  "payment_id": "pay_abc123",
  "order_id": "order_A1B2C3D4E5F6G7H8"
}
```

---

### Utility Endpoints

#### `GET /api/health`

Health check endpoint (no authentication required).

**Response (200):**
```json
{
  "status": "healthy",
  "timestamp": "2026-02-25T18:30:00.000000",
  "version": "2.0.0",
  "storage_backend": "pinecone",
  "vector_search_enabled": true
}
```

---

#### `POST /api/contact`

Handle contact form submissions.

**Request Body:**
```json
{
  "name": "Jane",
  "email": "jane@example.com",
  "subject": "Feedback",
  "message": "Great app!"
}
```

---

## 7. Data Models & Schema

### Memory Object

The canonical memory object returned by all API endpoints:

```typescript
interface Memory {
  id:                number;       // Unique identifier (timestamp-based)
  title:             string;       // Auto-generated title (first 5 words)
  content:           string;       // Full processed text content
  voice_text:        string;       // Original voice input
  category:          string;       // One of 8 predefined categories
  context:           string;       // Generated context description
  tags:              string[];     // AI-generated or keyword-based tags
  image_path:        string;       // Server path to uploaded image
  image_description: string;       // Gemini-generated image description
  created_at:        string;       // ISO 8601 timestamp
  updated_at:        string;       // ISO 8601 timestamp
  similarity_score?: number;       // Only present in search results (0.0–1.0)
  is_pdf_chunk?:     boolean;      // True if result is from a PDF chunk
  page_number?:      number;       // PDF page number (if applicable)
  snippet?:          string;       // PDF chunk preview text
}
```

### User Object

```typescript
interface User {
  id:            number;    // Auto-increment (SQLite)
  name:          string;
  email:         string;    // Unique
  password_hash: string;    // Werkzeug-generated hash
  memory_count:  number;    // Total memories saved
  is_premium:    boolean;   // Premium subscription status
  created_at:    string;    // ISO timestamp
}
```

### Categories

The system supports 8 predefined memory categories:

| # | Category | Trigger Keywords |
|---|---|---|
| 1 | Daily Life | home, family, personal, daily, today, morning, evening |
| 2 | Work & Meetings | work, meeting, office, colleagues, project, presentation |
| 3 | Learning & Growth | learn, study, education, growth, improve, knowledge |
| 4 | Health & Fitness | health, exercise, fitness, diet, wellness, medical |
| 5 | Money & Shopping | money, buy, purchase, shop, price, budget, finance |
| 6 | Entertainment & Leisure | movie, music, game, fun, relax, entertainment |
| 7 | Ideas & Creativity | idea, creative, innovation, design, think, brainstorm |
| 8 | General | Default (no keywords matched) |

---

## 8. Authentication & Authorization

### Authentication Flow

```
Client                          Server
  │                               │
  │  POST /api/register           │
  │  {name, email, password}      │
  │ ─────────────────────────────►│ → generate_password_hash(password)
  │                               │ → INSERT INTO users (SQLite)
  │  ◄──── 201 Created ──────────│
  │                               │
  │  POST /api/login              │
  │  {email, password}            │
  │ ─────────────────────────────►│ → get_user_by_email(email)
  │                               │ → check_password_hash(stored, input)
  │  ◄──── 200 + user object ────│
  │                               │
  │  (Client stores user.id in    │
  │   sessionStorage)             │
  │                               │
  │  GET /api/search-memories     │
  │  X-User-Id: {user.id}        │
  │ ─────────────────────────────►│ → @login_required validates header
  │  ◄──── 200 + results ────────│
```

### Key Design Decisions

1. **No JWT / Session Tokens:** Authentication is header-based (`X-User-Id`). The user ID is stored in `sessionStorage` on the client.
2. **Password Hashing:** Uses Werkzeug's `generate_password_hash` / `check_password_hash` (PBKDF2 by default).
3. **Google-Style Auth:** The `/api/auth/google` endpoint simulates Google account selection. It auto-creates users if they don't exist.
4. **User Scoping:** All memory operations filter by `user_id` in Pinecone metadata, ensuring data isolation.

---

## 9. AI / ML Pipeline

### Models Used

| Model | Provider | Dimension | Purpose |
|---|---|---|---|
| `gemini-2.5-flash` | Google | — | Image description, tag generation, memory summaries |
| `all-MiniLM-L6-v2` | HuggingFace | 384 | Text embedding for vector search |
| `Salesforce/blip-image-captioning-base` | HuggingFace | — | Fallback image captioning |
| `openai/whisper-tiny.en` | HuggingFace | — | Legacy speech-to-text (in processor.py) |
| `gemini-2.0-flash` | Google | — | Legacy categorization (in processor.py) |

### Processing Pipeline Detail

```python
def process_memory_logic(voice_text, image_path=None):
    """
    1. Image Analysis: If image provided, send to Gemini Vision
    2. Text Assembly: Combine voice_text + image description
    3. Tag Generation: Ask Gemini for 3-5 tags → fallback to keyword engine
    4. Category Classification: Keyword-based matcher → 8 categories
    5. Context Generation: Build descriptive context string
    6. Title Generation: First 5 words → "..."
    """
```

### Embedding Pipeline

```python
# Model: all-MiniLM-L6-v2
# Dimension: 384
# Normalization: L2 normalized (normalize_embeddings=True)

# Search text composition:
search_text = f"{content} {context} {' '.join(tags)}"
embedding = embedder.encode([search_text], normalize_embeddings=True)[0]

# Stored as: List[float] in Pinecone vector field
```

---

## 10. Vector Search Architecture

### Search Flow

```
User Query: "What was my project idea from last week?"
          │
          ▼
    ┌─────────────────────┐
    │ Generate Embedding  │ ← all-MiniLM-L6-v2
    │ (384 dimensions)    │
    └──────────┬──────────┘
               │
          ▼
    ┌─────────────────────┐
    │ Pinecone Query      │
    │ top_k=10            │
    │ threshold≥0.3       │
    │ filter: user_id     │ ← Supports $in for multi-ID (legacy migration)
    │ namespace: memories │
    └──────────┬──────────┘
               │
          ▼
    ┌─────────────────────┐
    │ Deduplicate         │ ← Remove duplicate memory IDs from chunk matches
    │ Fetch full metadata │
    │ Enrich with details │
    └──────────┬──────────┘
               │
          ▼
    ┌─────────────────────┐
    │ Text Fallback       │ ← Only if <5 vector results
    │ (in-memory filter   │    Fetches all memories, substring match
    │  on metadata)       │
    └──────────┬──────────┘
               │
          ▼
    ┌─────────────────────┐
    │ Merge & Sort        │ ← By similarity_score DESC
    │ Return Results      │
    └─────────────────────┘
```

### Similarity Metric

- **Metric:** Cosine Similarity
- **Threshold:** 0.3 (minimum score to include in results)
- **Top-K:** 10 for search, 5 for related memories

### Multi-User Scoping

All Pinecone queries include a `user_id` filter:

```python
# Single user:
filter_dict = {"user_id": {"$eq": 1}}

# Legacy migration (multiple IDs):
filter_dict = {"user_id": {"$in": [1, 1740567890]}}
```

---

## 11. Frontend Architecture

### Pages

| Page | File | Description |
|---|---|---|
| Landing / Demo | `templates/index.html` + `static/css/styles.css` + `static/js/script.js` | Marketing page with interactive demo |
| Dashboard | `index.html` + `styles.css` + `script.js` | Main app interface |
| Login | `login.html` | Login form + Google account selector |
| Register | `register.html` | Registration form |
| Contact | `contact.html` | Contact form |

### Client-Side Features (`script.js` — 1,630 lines)

| Feature | Implementation |
|---|---|
| **Speech Recognition** | Web Speech API (`webkitSpeechRecognition`) with continuous mode, silence detection with auto-stop timeout |
| **Image Capture** | File upload + camera capture (front/back toggle) via `getUserMedia` |
| **PDF Upload** | File input with base64 encoding for server transmission |
| **Memory Dashboard** | Dynamic rendering of memory cards with category badges, tags, images |
| **Search** | Real-time search with category/type filters |
| **Usage Tracking** | Progress bar showing memories used vs. limit |
| **Payment Modal** | Simulated payment flow with order display |
| **Theme Toggle** | Dark/Light mode switching with `data-theme` attribute |
| **Navigation** | Hamburger menu for mobile, smooth scroll to sections |
| **Session Management** | `sessionStorage`-based auth state |

### State Management

```javascript
// Auth state: stored in sessionStorage
sessionStorage.setItem('user', JSON.stringify(userObject));

// API calls include auth header:
headers: {
  'Content-Type': 'application/json',
  'X-User-Id': user.id
}
```

---

## 12. Configuration & Environment

### Environment Variables (`.env`)

| Variable | Required | Default | Description |
|---|---|---|---|
| `GEMINI_API_KEY` | Optional | — | Google Gemini API key for AI features |
| `PINECONE_API_KEY` | **Required** | — | Pinecone API key for vector database |
| `PINECONE_INDEX_NAME` | Optional | `memo-ai-index` | Pinecone index name |
| `FLASK_ENV` | Optional | `development` | Flask environment |
| `FLASK_DEBUG` | Optional | `True` | Flask debug mode |
| `SECRET_KEY` | Optional | — | Flask secret key for sessions |

### Application Constants

| Constant | Value | Location | Description |
|---|---|---|---|
| `MEMORY_FREE_LIMIT` | 10 | `app.py:61` | Free tier memory limit |
| `PREMIUM_PRICE_INR` | 299 | `app.py:522` | Premium price in INR |
| `PREMIUM_PRICE_USD` | 4.99 | `app.py:523` | Premium price in USD |
| `ALLOWED_EXTENSIONS` | `{png, jpg, jpeg, gif, pdf}` | `app.py:69` | Supported upload formats |
| `UPLOAD_FOLDER` | `./uploads/` | `app.py:68` | Upload directory |
| `DB_PATH` | `./memoai.db` | `models.py:20` | SQLite database path |

### Server Configuration

```python
app.run(
    host='0.0.0.0',
    port=5000,
    debug=True,
    threaded=True
)
```

---

## 13. Deployment

### Local Development

```bash
# 1. Clone the repository
git clone <repository-url>
cd memo-ai

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 5. Run the server
python app.py
# Server runs at http://localhost:5000
```

### Production Deployment

The project includes a `Procfile` for Heroku-style deployment:

```
web: unicorn app:app
```

> **Note:** The Procfile currently references `unicorn` instead of `gunicorn`. For production, use:
> ```
> web: gunicorn --bind 0.0.0.0:$PORT app:app
> ```

### Production Checklist

- [ ] Set `FLASK_DEBUG=False` and `FLASK_ENV=production`
- [ ] Set a strong `SECRET_KEY`
- [ ] Configure Pinecone API key for production index
- [ ] Configure Gemini API key with appropriate quotas
- [ ] Use `gunicorn` instead of Flask dev server
- [ ] Enable HTTPS
- [ ] Set up proper logging (replace `print()` with `logging`)
- [ ] Configure rate limiting with Redis backend (not in-memory)
- [ ] Set up database backups for SQLite
- [ ] Configure CORS with specific origins (not `*`)

---

## 14. Security Considerations

### Current Security Measures

| Area | Implementation |
|---|---|
| **Password Storage** | Werkzeug `generate_password_hash` (PBKDF2) |
| **Image Sanitization** | EXIF metadata stripped via Pillow before storage |
| **Input Validation** | Required fields checked on all endpoints |
| **Rate Limiting** | Flask-Limiter: 200/day, 50/hour default; per-endpoint overrides |
| **User Data Isolation** | Pinecone queries always filtered by `user_id` |
| **Secrets Management** | API keys in `.env` file (not committed to git) |
| **File Upload Restrictions** | Extension whitelist: `png, jpg, jpeg, gif, pdf` |

### Security Considerations for Production

| Concern | Current State | Recommendation |
|---|---|---|
| **Authentication** | Header-based (`X-User-Id`) | Implement JWT or session-based tokens |
| **CORS** | Allow all origins | Restrict to specific domains |
| **HTTPS** | Not enforced | Enable TLS/SSL |
| **Rate Limiting Storage** | In-memory | Use Redis or external store |
| **SQL Injection** | Parameterized queries (safe) | ✅ Already handled |
| **File Upload** | Extension check only | Add file content type validation |
| **Payment Verification** | Simulated (no signature verification) | Enable Razorpay/Stripe signature verification |
| **Error Messages** | Stack traces in console | Sanitize error responses in production |

---

## 15. Rate Limiting & Monetization

### Rate Limits

| Scope | Limit | Key Function |
|---|---|---|
| Global Default | 200/day, 50/hour | `get_user_id_as_key()` (user ID or IP) |
| Search Endpoint | 50/day | Per user |
| Summary Endpoint | 10/day | Per user |
| Storage | In-memory (`memory://`) | Resets on server restart |

### Freemium Model

```
┌─────────────────────────────────────┐
│           FREE TIER                 │
│  • 10 memories per account          │
│  • All features available           │
│  • Returns HTTP 402 when limit hit  │
│    (with payment_required flag)     │
└──────────────┬──────────────────────┘
               │  Payment
               ▼
┌─────────────────────────────────────┐
│         PREMIUM TIER                │
│  • Unlimited memories               │
│  • ₹299/month (INR)                 │
│  • $4.99/month (USD)                │
│  • is_premium=1 in SQLite           │
└─────────────────────────────────────┘
```

### Payment Flow

```
1. Client detects limit → shows payment modal
2. POST /api/payment/initiate → returns simulated order_id
3. Client shows payment UI with order details
4. Client submits payment → POST /api/payment/verify
5. Server sets is_premium=1 in SQLite
6. Client receives confirmation, limit removed
```

---

## 16. Error Handling & Logging

### Error Handling Strategy

- All route handlers are wrapped in try/except blocks
- Errors return standardized JSON:
  ```json
  { "error": "Error type", "message": "Description" }
  ```
- Status codes follow HTTP conventions:
  - `400` — Bad request / validation error
  - `401` — Authentication required  
  - `402` — Payment required  
  - `409` — Conflict (duplicate email)  
  - `429` — Rate limit exceeded  
  - `500` — Internal server error

### Logging

Currently uses `print()` statements for logging:

```python
print(f"Error processing memory: {e}")
import traceback
traceback.print_exc()
```

Key log patterns:
- `[Auth]` prefix — User authentication operations
- `Pinecone:` prefix — Pinecone connection status
- Service initialization messages at startup

---

## 17. Known Limitations & Future Work

### Limitations

| Area | Limitation |
|---|---|
| **Authentication** | Header-based; no token validation or session expiry |
| **Speech Recognition** | Browser-dependent (Web Speech API); no server-side ASR in production flow |
| **Memory Metadata** | Pinecone metadata limited to ~40KB; content truncated to 1000 chars |
| **PDF Chunks** | Max 100 chunks per memory; deletion deletes all 100 IDs regardless |
| **Text Search Fallback** | Loads all memories into memory for substring matching (O(n)) |
| **Pinecone Query Limit** | `get_all_memories` uses `top_k=1000` — won't return more than 1000 memories |
| **Payment** | Simulated only; no real payment processing |
| **Procfile** | Contains `unicorn` instead of `gunicorn` |
| **Legacy Code** | `backend/processor.py` contains FAISS/Whisper code that's partially superseded |

### Future Enhancements

- [ ] JWT-based authentication with refresh tokens
- [ ] Server-side speech-to-text with Whisper API
- [ ] Real payment integration (Razorpay / Stripe)
- [ ] Memory sharing and collaboration
- [ ] Export/import functionality
- [ ] Proper logging framework (Python `logging` module)
- [ ] Pagination for large memory collections
- [ ] WebSocket support for real-time updates
- [ ] Automated testing suite
- [ ] CI/CD pipeline
- [ ] Docker containerization

---

> **Document Generated:** February 25, 2026  
> **Project Version:** 2.0.0  
> **Status:** Active Development
