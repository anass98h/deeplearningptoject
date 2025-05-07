import uvicorn
import os
import sys


sys.path.append(os.path.dirname(os.path.abspath(__file__)))


from video_processing_route import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    print(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)