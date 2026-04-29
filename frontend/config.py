# frontend/config.py
import os

# In development this points to your local FastAPI server.
# In production (Streamlit Cloud) this will be your Render URL.
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")