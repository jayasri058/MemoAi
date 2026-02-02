# MemoAI - Intelligent Personal Memory Assistant

![MemoAI Logo](https://placehold.co/400x200/4f46e5/white?text=MemoAI)

MemoAI is an intelligent personal memory assistant that helps you capture, categorize, and retrieve your thoughts, conversations, and visual memories with ease.

## 🌟 Features

- **🎤 Voice Recognition**: Convert spoken thoughts to text instantly
- **🧠 Smart Categorization**: Automatically classify content into meaningful categories
- **🖼️ Visual Memory**: Store and retrieve visual memories with context
- **🔍 Intelligent Search**: Find memories using natural language queries
- **💾 Local Storage**: Secure offline storage of your memories
- **📱 Responsive Design**: Works on all devices

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- Node.js (for development server)
- Google Gemini API Key (optional for enhanced features)

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd memo-ai
```

2. **Create virtual environment:**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables:**
```bash
cp .env.example .env
# Edit .env and add your API keys
```

5. **Run the application:**
```bash
python app.py
```

6. **Access the application:**
Open your browser and navigate to `http://localhost:5000`

## 📁 Project Structure

```
memo-ai/
├── app.py              # Flask backend server
├── index.html          # Main HTML interface
├── script.js           # Frontend JavaScript logic
├── styles.css          # Styling and animations
├── requirements.txt    # Python dependencies
├── .env.example       # Environment variables template
├── .gitignore         # Git ignore rules
└── uploads/           # Directory for uploaded images
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
# Google Gemini API Key (optional)
GEMINI_API_KEY=your_actual_api_key_here

# Flask Configuration
FLASK_ENV=development
FLASK_DEBUG=True

# Secret Key for sessions
SECRET_KEY=your_secret_key_here
```

### API Endpoints

- `POST /api/process-memory` - Process voice and image input
- `GET /api/search-memories?q=query` - Search stored memories  
- `GET /api/health` - Health check endpoint

## 🛠️ Development

### Running Tests

```bash
# Run unit tests
python -m pytest tests/

# Run with coverage
python -m pytest --cov=app tests/
```

### Building for Production

```bash
# Install build tools
pip install gunicorn

# Run with Gunicorn
gunicorn --bind 0.0.0.0:5000 app:app
```

## 🔒 Security

- All API keys are stored in environment variables
- User data is stored locally in browser storage
- Images are sanitized before processing
- Input validation on all endpoints

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Google Gemini AI for natural language processing
- Whisper ASR for speech recognition
- FAISS for vector similarity search
- Sentence Transformers for embedding generation

## 📞 Support

For support, email support@memoai.app or join our [Discord community](https://discord.gg/memoai).

---

Made with ❤️ by the MemoAI Team