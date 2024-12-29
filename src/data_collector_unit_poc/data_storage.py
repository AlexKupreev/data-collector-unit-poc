import datetime
from enum import Enum

from sqlalchemy import Column, String, Integer, JSON, DateTime, Enum as SqlEnum
from sqlalchemy.orm import sessionmaker, Mapped, mapped_column

from data_collector_unit_poc.core import db


class Source(Enum):
    HACKER_NEWS = "Hacker News"
    LOBSTERS = "Lobsters"

class Post(db.Base):
    __tablename__ = 'posts'

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    original_id: Mapped[str] = mapped_column(String, unique=False, nullable=True)
    title: Mapped[str] = mapped_column(String, nullable=False)
    url: Mapped[str] = mapped_column(String, nullable=False)
    source: Mapped[Source] = mapped_column(SqlEnum(Source), nullable=False)
    score: Mapped[int] = mapped_column(Integer, nullable=True)
    timestamp: Mapped[datetime.datetime] = mapped_column(DateTime, nullable=False)
    content: Mapped[str] = mapped_column(String, nullable=False)
    comments: Mapped[dict] = mapped_column(JSON, nullable=True)
    description: Mapped[str] = mapped_column(String, nullable=False)
    document_uid: Mapped[str] = mapped_column(String, nullable=False)
    ingest_utctime: Mapped[int] = mapped_column(Integer, nullable=False)


# class User(db.Base):
#     __tablename__ = "users"

#     id = Column(Integer, primary_key=True)
#     name = Column(String)

class PostRepository:
    def __init__(self, db_engine=db.db_engine):
        self.Session = sessionmaker(bind=db_engine)

    def add_post(self, post_data):
        session = self.Session()
        post = Post(
            original_id=post_data['original_id'],
            title=post_data['title'],
            url=post_data['url'],
            source=Source(post_data['source']),
            score=post_data.get('score'),
            timestamp=datetime.datetime.fromisoformat(post_data['timestamp']),
            content=post_data['content'],
            comments=post_data.get('comments'),
            description=post_data['description'],
            document_uid=post_data['document_uid'],
            ingest_utctime=post_data['ingest_utctime']
        )
        session.add(post)
        session.commit()
        session.close()

    def get_post_by_id(self, original_id):
        session = self.Session()
        post = session.query(Post).filter_by(original_id=original_id).first()
        session.close()
        return post

    def delete_post(self, original_id):
        session = self.Session()
        post = session.query(Post).filter_by(original_id=original_id).first()
        if post:
            session.delete(post)
            session.commit()
        session.close()
