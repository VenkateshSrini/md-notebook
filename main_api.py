from dotenv import load_dotenv
load_dotenv()

import os
import uvicorn
from notebook_api.app import app

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
