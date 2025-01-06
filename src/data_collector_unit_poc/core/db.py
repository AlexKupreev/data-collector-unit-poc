"""Main DB configuration - entry point for all DB operations"""
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase

# Determine the environment
environment = os.getenv("ENVIRONMENT", "production")

# Set the database path based on the environment
if environment == "production":
    db_url = "sqlite:////data/posts.db"
else:
    db_url = f"sqlite:///{os.path.join(os.path.dirname(__file__), '..', 'migrations', 'posts.db')}"

class Base(DeclarativeBase):
    pass

db_engine = create_engine(db_url, connect_args={"check_same_thread": False})
# Base.metadata.create_all(db_engine)
