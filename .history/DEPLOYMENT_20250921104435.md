# Deep Researcher - Deployment Guide

## Render Deployment

### Backend Service Configuration

**Service Type**: Web Service

**Build Command**: 
```
pip install -r requirements.txt
```

**Start Command**: 
```
python api_server_final.py
```

**Environment Variables**:
- `PORT`: Automatically set by Render
- `PYTHON_VERSION`: `3.9` or `3.10`

### Frontend Service Configuration (Optional)

**Service Type**: Static Site

**Build Command**:
```
cd frontend && npm install && npm run build
```

**Publish Directory**: `frontend/build`

### Alternative: Single Service Deployment

If you want to serve both backend and frontend from one service:

1. **Build Command**:
```
pip install -r requirements.txt && cd frontend && npm install && npm run build
```

2. **Start Command**:
```
python api_server_final.py
```

3. **Add static file serving** to your FastAPI app (see below)

## Environment Setup

### Required Files
- `requirements.txt` - Python dependencies
- `api_server_final.py` - Main application
- `deep_researcher/` - Core application modules
- `data/` - Database and vector storage (will be created)

### Optional Files
- `frontend/` - React frontend (if deploying separately)
- `sample_documents/` - Sample documents for testing

## Static File Serving (Optional)

To serve the React frontend from the same service, add this to your FastAPI app:

```python
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Serve static files
app.mount("/static", StaticFiles(directory="frontend/build/static"), name="static")

@app.get("/{full_path:path}")
async def serve_react_app(full_path: str):
    if full_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="API endpoint not found")
    return FileResponse("frontend/build/index.html")
```

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Ensure all dependencies are in `requirements.txt`
2. **Port binding errors**: The app now uses `os.environ.get("PORT", 9000)`
3. **Memory issues**: Consider upgrading to a higher tier plan for ML models
4. **Build timeouts**: The embedding model download might take time

### Performance Tips

1. Use `requirements-prod.txt` for production (excludes dev dependencies)
2. Consider using a persistent disk for the `data/` directory
3. Monitor memory usage with the ML models

## Health Check

The API provides a health check endpoint:
- `GET /health` - Returns system status
- `GET /status` - Returns detailed system information

## Sample Deployment URLs

After deployment, your API will be available at:
- `https://your-app-name.onrender.com/health`
- `https://your-app-name.onrender.com/research` (POST)
- `https://your-app-name.onrender.com/suggest` (POST)
