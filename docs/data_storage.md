# Data Storage

Post data is stored in a local sqLite database.
Database operations are performed via sqlAlchemy ORM.
The code implements Repository pattern allowing to decouple the model layer from the data layer.

As for now fields of the data structure can be represented as a plain json structure like:

```json
[
    {
        "original_id": "tbcweu",
        "title": "The Zero Click Internet",
        "url": "https://www.techspot.com/article/2908-the-zero-click-internet/",
        "score": 1,
        "timestamp": "2024-10-27T06:02:49.000-05:00",
        "source": "Lobsters",
        "content": "The internet is in the midst of undergoing the biggest change since its inception. It's huge. And there is no going back. The web is changing into the Zero Click Internet, and it will change everything about how you do everything. Zero Click Internet means you'll no longer click on links to find the content you want.",
        "comments": [
        {
            "author": "dpk",
            "text": "<p>I think this article is somewhat alarmist and the future it portrays not nearly as inevitable as it claims. (Although consider that if you are a regular Lobsters reader, the way you use the internet is probably already atypical in many ways, especially if you have been around the internet a long time.)</p>",
            "time": "2024-10-27T06:31:45.000-05:00"
        },
        {
            "author": "symgryph",
            "text": "<p>I do find the llm summaries. Very annoying. It might be that gets to a point where we have to start indexing things ourselves! I also find search engines less and less relevant as I go directly to things like say GitHub, or medical sites directly since the search engine results are becoming more and more crappy.</p>\n",
            "time": "2024-10-27T07:08:04.000-05:00"
        },
        ],
        "description": "The internet is undergoing the biggest change since its inception. It's huge. And there is no going back. The web is changing into the Zero Click Internet,...",
        "document_uid": "f8838702cd",
        "ingest_utctime": 1730039398
    },
]
```
