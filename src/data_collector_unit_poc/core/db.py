"""Main DB configuration - entry point for all DB operations"""
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase

# Construct the absolute path to the database file
db_url = f"sqlite:///{os.path.join(os.path.dirname(__file__), '..', 'migrations', 'posts.db')}"

class Base(DeclarativeBase):
    pass

db_engine = create_engine(db_url, connect_args={"check_same_thread": False})
# Base.metadata.create_all(db_engine)
