import hashlib
import httpx
import time
from datetime import datetime
from pathlib import Path
import cleanurl
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm.auto import tqdm
from bs4 import BeautifulSoup
from readability import Document
from data_collector_unit_poc.data_storage import PostRepository, Source
from data_collector_unit_poc.notebooks import embeddings
from data_collector_unit_poc.notebooks import vectordb

class ContentIsVideoError(Exception):
    pass

def generate_document_id(url: str | None, content: str | None) -> str:
    """Generate unique document identifier based on URL or cleaned content"""
    MAX_CHARS = 240
    if url:
        cleaned = cleanurl.cleanurl(url.strip())
        cleaned_url = getattr(cleaned, 'schemeless_url', url) or getattr(cleaned, 'url', url)
    elif content:
        cleaned_url = content.strip().lower()
    else:
        raise ValueError("URL and Content is empty")
    truncated = cleaned_url[:MAX_CHARS]
    hash_object = hashlib.sha1(truncated.encode())
    hash_hex = hash_object.hexdigest()
    document_id = hash_hex[:10]
    return document_id

def fetch_url_content(client, url, truncate_words: int = 500) -> str:
    if "youtube.com" in url or "youtu.be" in url:
        raise ContentIsVideoError("YouTube video content is not available.")
    response = client.get(url, follow_redirects=True)
    doc = Document(response.content)
    soup = BeautifulSoup(doc.summary(), 'html.parser')
    for script in soup(["script", "style"]):
        script.decompose()
    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)
    if truncate_words:
        words = text.split()[:truncate_words]
        return ' '.join(words)
    return text

def fetch_meta_description(client, url):
    try:
        response = client.get(url, follow_redirects=True, timeout=10.0)
        soup = BeautifulSoup(response.text, 'html.parser')
        meta_desc = soup.find('meta', attrs={'name': 'description'}) or soup.find('meta', attrs={'property': 'og:description'})
        return meta_desc['content'] if meta_desc else "No description available."
    except Exception as e:
        return f"Error fetching meta description: {str(e)}"

def fetch_wayback_content(client, url, truncate_words: int = 500):
    try:
        if "youtube.com" in url or "youtu.be" in url:
            raise ContentIsVideoError("YouTube video content is not available.")
        wb_url = f"http://archive.org/wayback/available?url={url}"
        response = client.get(wb_url, follow_redirects=True)
        data = response.json()
        if "archived_snapshots" in data and "closest" in data["archived_snapshots"]:
            snapshot_url = data["archived_snapshots"]["closest"]["url"]
            response = client.get(snapshot_url)
            soup = BeautifulSoup(response.text, 'html.parser')
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            words = text.split()[:truncate_words]
            return ' '.join(words)
        else:
            print("No Wayback Machine snapshot available.")
            return None
    except Exception as e:
        print(f"Error fetching Wayback content: {str(e)}")
        return None

def fetch_hn_comments(client, item_id, max_comments=10):
    url = f"https://hacker-news.firebaseio.com/v0/item/{item_id}.json"
    response = client.get(url)
    item_data = response.json()
    comments = []
    if "kids" in item_data:
        for comment_id in item_data["kids"][:max_comments]:
            comment_url = f"https://hacker-news.firebaseio.com/v0/item/{comment_id}.json"
            comment_response = client.get(comment_url)
            comment_data = comment_response.json()
            if comment_data.get("text"):
                comments.append({
                    "author": comment_data.get("by", "Anonymous"),
                    "text": comment_data["text"],
                    "time": datetime.fromtimestamp(comment_data["time"]).isoformat()
                })
    return comments

def fetch_lobsters_comments(client, short_id, max_comments=10):
    url = f"https://lobste.rs/s/{short_id}.json"
    response = client.get(url)
    story_data = response.json()
    comments = []
    for comment in story_data.get("comments", [])[:max_comments]:
        comments.append({
            "author": comment.get("commenting_user", {}),
            "text": comment["comment"],
            "time": comment["created_at"]
        })
    return comments

def get_stories_to_download(stories: list, repository: PostRepository, source: str) -> list:
    """Get list of stories for a source that need to be downloaded"""
    stories_to_download = []
    if source == "Hacker News":
        story_ids = [str(story_id) for story_id in stories]
    elif source == "Lobsters":
        story_ids = [str(story["short_id"]) for story in stories]
    else:
        raise ValueError("Unknown source")
    
    for story_id in story_ids:
        if not repository.get_post_by_id(story_id):
            stories_to_download.append(story_id)
    return stories_to_download

def fetch_hacker_news(scope: str, repository: PostRepository, count: int = 10, max_comments: int = 10, truncate_words: int = 1000) -> list[dict]:
    with httpx.Client() as client:
        if scope == "hottest":
            url = "https://hacker-news.firebaseio.com/v0/topstories.json"
        elif scope == "newest":
            url = "https://hacker-news.firebaseio.com/v0/newstories.json"
        else:
            raise ValueError(f"Unknown scope: {scope}")
        response = client.get(url)
        stories = response.json()[:count]
        stories_to_download = get_stories_to_download(stories, repository=repository, source="Hacker News")
        news_items = []
        for story_id in stories_to_download:
            story_url = f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json"
            story_response = client.get(story_url)
            story_data = story_response.json()
            item_url = story_data.get("url", f"https://news.ycombinator.com/item?id={story_id}")
            meta_description = fetch_meta_description(client, item_url)
            content = story_data.get("text")
            if not content and item_url:
                try:
                    content = fetch_wayback_content(client, item_url, truncate_words=truncate_words)
                    if not content:
                        content = fetch_url_content(client, item_url, truncate_words=truncate_words)
                except ContentIsVideoError:
                    pass
                except Exception as exc:
                    pass
                if not content:
                    content = story_data.get("title")
            comments = fetch_hn_comments(client, story_id, max_comments)
            news_item = {
                "original_id": story_id,
                "title": story_data["title"],
                "url": item_url,
                "score": story_data["score"],
                "timestamp": datetime.fromtimestamp(story_data["time"]).isoformat(),
                "source": "Hacker News",
                "content": content,
                "comments": comments,
                "description": meta_description,
                "document_uid": generate_document_id(item_url, content),
                "ingest_utctime": int(time.time())
            }
            repository.add_post(news_item)
            news_items.append(news_item)
        return news_items

def fetch_lobsters_news(scope: str, repository: PostRepository, count: int = 10, max_comments: int = 10, truncate_words: int = 1000) -> list[dict]:
    with httpx.Client() as client:
        if scope == "hottest":
            url = "https://lobste.rs/hottest.json"
        elif scope == "newest":
            url = "https://lobste.rs/newest.json"
        else:
            raise ValueError(f"Unknown scope: {scope}")
        response = client.get(url)
        stories = response.json()[:count]
        stories_to_download = get_stories_to_download(stories, repository=repository, source="Lobsters")
        news_items = []
        for story in stories:
            if str(story["short_id"]) not in stories_to_download:
                continue
            content = None
            meta_description = story.get("description", "")
            if meta_description:
                content = meta_description
            else:
                meta_description = fetch_meta_description(client, story["url"])
            comments = fetch_lobsters_comments(client, story["short_id"], max_comments)
            if not content and story["url"]:
                try:
                    content = fetch_wayback_content(client, story["url"], truncate_words=truncate_words)
                    if not content:
                        content = fetch_url_content(client, story["url"], truncate_words=truncate_words)
                except ContentIsVideoError:
                    pass
                except Exception as exc:
                    print(f"Error fetching content for {story['url']}: {str(exc)}, fallback to Wayback Machine.")
            if not content:
                content = story["title"]
            news_item = {
                "original_id": story["short_id"],
                "title": story["title"],
                "url": story["url"],
                "score": story["score"],
                "timestamp": story["created_at"],
                "source": "Lobsters",
                "content": content,
                "comments": comments,
                "description": meta_description,
                "document_uid": generate_document_id(story["url"], content),
                "ingest_utctime": int(time.time())
            }
            repository.add_post(news_item)
            news_items.append(news_item)
        return news_items

def main():
    load_dotenv(Path() / "../.env", verbose=True)
    INIT_CHUNK_SIZE = 100
    ONGOING_CHUNK_SIZE = 20
    repository = PostRepository()
    is_first_run = not repository.get_post_by_id("1")  # Assuming "1" is a valid ID to check for first run
    hn_news = fetch_hacker_news(
        scope="newest",
        count=INIT_CHUNK_SIZE if is_first_run else ONGOING_CHUNK_SIZE,
        repository=repository
    )
    print(f"Number of downloaded HN news: {len(hn_news)}")
    lr_news = fetch_lobsters_news(
        scope="newest",
        count=INIT_CHUNK_SIZE if is_first_run else ONGOING_CHUNK_SIZE,
        repository=repository
    )
    print(f"Number of downloaded LR news: {len(lr_news)}")
    documents = hn_news + lr_news
    embedding_dim = embeddings.get_dimensions()
    operations = []
    for doc in tqdm(documents, desc="Creating embeddings"):
        doc["vector"] = embeddings.get_embeddings(doc["content"])
        operations.append(doc)
    MAX_CONTENT_VECTORIZED = 1000 * 5
    collection_name = "llm_summarizer_poc"
    collection_db_path = "./milvus_summarizer.db"
    milvus_client = vectordb.MilvusClientFix.get_instance(collection_db_path)
    if is_first_run and milvus_client.has_collection(collection_name):
        milvus_client.drop_collection(collection_name)
    milvus_client.create_collection(
        collection_name=collection_name,
        dimension=embedding_dim,
        schema=vectordb.create_schema("LLM Summarizer PoC", embedding_dim, MAX_CONTENT_VECTORIZED),
        index_params=vectordb.create_index_params(milvus_client),
        metric_type="IP",
        consistency_level="Strong",
    )
    operations = [
        {
            "document_uid": doc["document_uid"],
            "text": doc["content"],
            "ingest_utctime": doc["ingest_utctime"],
            "vector": doc["vector"]
        }
        for doc in operations
    ]
    milvus_client.insert(collection_name=collection_name, data=operations)
    milvus_client.close()
    completed_at = time.time()
    time_spent = (completed_at - started_at)
    print(f"Completed at {datetime.now()}, execution took ~{int(time_spent / 60)} min")

if __name__ == "__main__":
    started_at = time.time()
    main()
