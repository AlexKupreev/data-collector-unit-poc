from sqlalchemy import create_engine, Column, String, Integer, JSON, DateTime, Enum as SqlEnum
from sqlalchemy.orm import sessionmaker, DeclarativeBase, Mapped, mapped_column
from enum import Enum
import datetime

class Base(DeclarativeBase):
    pass

class Source(Enum):
    HACKER_NEWS = "Hacker News"
    LOBSTERS = "Lobsters"

class Post(Base):
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

class PostRepository:
    def __init__(self, db_url='sqlite:///posts.db'):
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

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
