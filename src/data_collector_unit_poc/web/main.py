"""Main app entrypoint"""
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqladmin import Admin, ModelView

from data_collector_unit_poc.core.db import db_engine
from data_collector_unit_poc.web import frontend, scheduler


app = FastAPI(lifespan=scheduler.lifespan)

# admin = Admin(app, db_engine)

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this to restrict origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
async def healthcheck():
    return {"status": "ok"}


frontend.init(app)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")
