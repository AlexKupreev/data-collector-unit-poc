"""Vector DB functionality"""
import os
from datetime import datetime

from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient

from embeddings import get_embeddings

class MilvusClientFix:
    """A wrapper for Milvus to deal with local issues."""
    @staticmethod
    def get_instance(filepath: str) -> MilvusClient:
        """Get the Milvus client instance."""
        abslockpath = os.path.abspath(filepath)
        lockdir, lockfile = os.path.split(abslockpath)
        lock_path = os.path.join(lockdir, f".{lockfile}.lock")
        if os.path.isfile(lock_path):
            os.remove(lock_path)

        return MilvusClient(uri=filepath)


def create_schema(description: str, embedding_dim: int, max_text_len: int):
    """Create the schema for the Milvus collection."""
    id_field = FieldSchema(name="document_uid", dtype=DataType.VARCHAR, is_primary=True, description="primary id", max_length=8)
    text_field = FieldSchema(name="text", dtype=DataType.VARCHAR, description="text", max_length=max_text_len)
    timestamp_field = FieldSchema(name="ingest_utctime", dtype=DataType.INT32, description="UTC timestamp of ingestion")
    vector_field = FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim, description="vector")

    # Set enable_dynamic_field to True if you need to use dynamic fields.
    milvus_schema = CollectionSchema(
        fields=[id_field, text_field, timestamp_field, vector_field],
        auto_id=False,
        enable_dynamic_field=True,
        description=description
    )

    return milvus_schema

def create_index_params(milvus_client: MilvusClient):
    """Create the index parameters for the Milvus collection."""
    index_params = milvus_client.prepare_index_params()

    index_params.add_index(
        field_name="document_uid",
        index_type="INVERTED"
    )

    index_params.add_index(
        field_name="ingest_utctime",
        index_type=""
    )

    # # indexes on scalar fields are needed to speed up filtering like in other DBs
    # index_params.add_index(
    #     field_name="course",
    #     index_type="INVERTED"
    # )

    index_params.add_index(
        field_name="vector",
        index_type="HNSW",  # for local search only HNSW is supported
        metric_type="IP",
        params={"nlist": 128}
    )

    return index_params


def milvus_search(
    milvus_client: MilvusClient,
    collection_name: str,
    query: str | list,
    num_results: int = 20,
    start_dt: datetime | None = None,
    end_dt: datetime | None = None,
) -> list[dict]:
    filters = None
    if start_dt is not None and end_dt is not None:
        filters = f"{int(start_dt.timestamp())} <= ingest_utctime <= {int(end_dt.timestamp())}"

    elif start_dt is not None:
        filters = f"ingest_utctime >= {int(start_dt.timestamp())}"

    elif end_dt is not None:
        filters = f"ingest_utctime <= {int(end_dt.timestamp())}"

    # if a list is passed, treat it as a vector and do not change it
    search_vector = query if isinstance(query, list) else get_embeddings(query)

    raw_results = milvus_client.search(
        collection_name=collection_name,
        data=[
            search_vector
        ],
        filter=filters,
        limit=num_results,  # Return top num_results results
        search_params={"metric_type": "IP", "params": {}},  # Inner product distance
        output_fields=["text", "document_uid", "vector"],  # Return the text field
    )

    results = [x['entity'] for x in list(raw_results)[0] if x["entity"]["vector"] != search_vector]

    return results
