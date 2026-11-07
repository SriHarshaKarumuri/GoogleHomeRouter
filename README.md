# Train Enquiry Assistant — Dialogflow Webhook API

FastAPI backend for Google Assistant / Dialogflow webhook requests with Gemini function calling and CT Trains API.

## Free Deployment

### Railway (Recommended — No Card Required)
1. Go to https://railway.com, sign up with GitHub
2. New Project → Deploy from GitHub repo
3. Settings → Variables → Add `GOOGLE_API_KEY` = your key
4. Settings → Networking → Generate Domain

Webhook URL: `https://your-app.up.railway.app/voice-router`

### Render (Free Tier — No Card)
1. Sign up at https://render.com with GitHub
2. New → Web Service → connect this repo
3. Environment: Python 3
4. Build: `pip install -r requirements.txt`
5. Start: `uvicorn main:app --host 0.0.0.0 --port $PORT`
6. Add env var: `GOOGLE_API_KEY`

URL: `https://your-app.onrender.com/voice-router`

## Local Setup

```bash
pip install -r requirements.txt
echo "GOOGLE_API_KEY=your_key_here" > .env
python -m uvicorn main:app --reload --port 8000
```

Endpoints: `POST /voice-router`, `GET /health`
