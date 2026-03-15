"""
main.py
-------
Entry point. Run with:  python main.py
Or for production:      uvicorn main:app --host 0.0.0.0 --port 8000
"""

import os
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)

from api.server import app   # noqa: E402  (import after logging setup)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=False,
        log_level="info",
    )
