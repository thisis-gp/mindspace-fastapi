# MindSpace Backend 🌐🧠

[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Online-green?style=for-the-badge&logo=vercel)](https://mind-space-phi.vercel.app/)
[![Frontend](https://img.shields.io/badge/Frontend-React-blue?style=for-the-badge&logo=react)](https://github.com/thisis-gp/MindSpace)

AI-powered backend for mental health companion application featuring multilingual support, emotional analysis, and Firebase integration.

## Key Features ✨

- **Gemini AI Integration**  
  🧠 Context-aware mental health conversations with Google's LLM
- **Multilingual Support**  
  🌍 10+ Indian languages with auto-detection (Hindi, Tamil, Bengali, etc.)
- **Real-time Database**  
  🔥 Firebase integration for user history and reports
- **Text-to-Speech**  
  🔊 Dynamic audio responses with language-specific accents
- **Safety Protocols**  
  🛡️ Content moderation and ethical AI guidelines
- **Severity Analysis**  
  📈 Symptom-based scoring system for mental health assessment

## Tech Stack 🛠️

**Core**  
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Firebase](https://img.shields.io/badge/Firebase-FFCA28?style=flat&logo=firebase&logoColor=black)](https://firebase.google.com/)

**AI Services**  
[![Gemini](https://img.shields.io/badge/Google%20Gemini-4285F4?style=flat&logo=google&logoColor=white)](https://deepmind.google/technologies/gemini/)
[![gTTS](https://img.shields.io/badge/gTTS-00C300?style=flat&logo=google-cloud&logoColor=white)](https://gtts.readthedocs.io/)

## Installation 💻

1. Clone repository
```bash
git clone https://github.com/thisis-gp/mindspace-fastapi.git
cd mindspace-fastapi
```
2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Configure environment variables (.env)
```ini
GEMINI_API_KEY=your_gemini_key
DATABASE_URL=your_firebase_url
SERVICE_ACCOUNT_KEY_PATH=./serviceAccountKey.json
```

4. Start server
```bash
uvicorn main:app --reload
```

## Frontend Integration 🔗
Connect with [React Frontend](https://github.com/thisis-gp/MindSpace)


