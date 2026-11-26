import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from .utils import *
from . import config

logger = get_logger()


def load_tokenized_data(statement: str = "account") -> pd.DataFrame:
    """
    Load tokenized transactions from DuckDB where trans_id is not null.
    """
    conn = create_duckdb_connection(config.DATABASE_PATH)
    df = conn.query(f"""
        SELECT *
        FROM {config.SCHEMA_NAME}.{config.tables(statement).tokens}
        WHERE trans_id IS NOT NULL
    """).to_df()

    logger.info(f"✅ Loaded {len(df)} tokenized records")
    return df


def _upsert_chunked(collection, ids, embeddings, metadatas, documents):
    # Chroma rejects single writes above its max batch size (~5461)
    max_bs = collection._client.get_max_batch_size()
    for i in range(0, len(ids), max_bs):
        j = i + max_bs
        collection.upsert(
            ids=ids[i:j],
            embeddings=[e.tolist() for e in embeddings[i:j]],
            metadatas=metadatas[i:j],
            documents=documents[i:j],
        )


def create_transaction_embeddings(model, df: pd.DataFrame, corpus_col: str, txn_collection, batch_size: int = 64):
    """
    Encode transaction-level embeddings in batches and store them in ChromaDB.
    Ensures embeddings and metadata stay aligned and persist correctly.
    """
    logger.info("Encoding transaction-level embeddings in batches...")

    texts, ids, metadatas = [], [], []

    for _, row in df.iterrows():
        tokens = row[corpus_col]
        if not isinstance(tokens, (list, np.ndarray)) or len(tokens) == 0:
            continue

        trans_id = row["trans_id"]
        if trans_id in ids:
            logger.warning(f"Duplicate transaction ID detected: {trans_id}, skipping.")
            continue

        text = " ".join(tokens)
        texts.append(text)
        ids.append(trans_id)
        metadatas.append({
            "trans_mode": row.get("trans_mode") or "Unknown",
            "trans_type": row.get("trans_type"),
            "trans_amt": row.get("trans_amt")
        })

    logger.info(f"Total transactions to encode: {len(texts)}")

    # Batch encode embeddings
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Transaction Batches"):
        batch_texts = texts[i:i + batch_size]
        batch_embeddings = model.encode(batch_texts, show_progress_bar=False)
        embeddings.extend(batch_embeddings)

    # Safety check
    assert len(embeddings) == len(ids) == len(texts) == len(metadatas), \
        "Embedding, ID, and metadata count mismatch!"

    _upsert_chunked(txn_collection, ids, embeddings, metadatas, texts)

    logger.info(f"✅ Added {len(texts)} transaction embeddings successfully")


def create_token_embeddings(model, df: pd.DataFrame, corpus_col: str, token_collection, batch_size: int = 128):
    """
    Encode token-level embeddings in batches and store them in ChromaDB.
    Ensures no duplicate token IDs.
    """
    logger.info("Encoding token-level embeddings in batches...")

    token_records = []
    seen_ids = set()
    duplicate_ids = set()

    # Flatten tokens with metadata
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Preparing Tokens"):
        for tok in row[corpus_col]:
            if not tok:
                logger.warning(f"Empty token for transaction ID {row['trans_id']}, skipping.")
                continue

            trans_id = row['trans_id']
            token_id = f"{trans_id}_{tok}"

            if token_id in seen_ids:
                logger.warning(f"Duplicate token ID detected: {token_id}, skipping.")
                duplicate_ids.add(trans_id)
                continue

            token_records.append({
                "id": token_id,
                "token": tok,
                "trans_id": trans_id
            })
            seen_ids.add(token_id)

    texts = [r["token"] for r in token_records]
    ids = [r["id"] for r in token_records]
    metadatas = [{"trans_id": r["trans_id"]} for r in token_records]

    logger.info(f"Total tokens to encode: {len(texts)}")

    # Batch encode embeddings
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Token Batches"):
        batch_texts = texts[i:i + batch_size]
        batch_embeddings = model.encode(batch_texts, show_progress_bar=False)
        embeddings.extend(batch_embeddings)

    _upsert_chunked(token_collection, ids, embeddings, metadatas, texts)

    logger.info(f"✅ Added {len(token_records)} token embeddings")


def main(statement: str = "account"):
    t = config.tables(statement)
    # Setup ChromaDB client
    logger.info("Creating ChromaDB client...")
    chroma_client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)

    # Get collections
    txn_collection = chroma_client.get_or_create_collection(t.txn_collection)
    token_collection = chroma_client.get_or_create_collection(t.token_collection)

    # Load embedding model
    logger.info("Loading Sentence Transformer model...")
    model = SentenceTransformer(config.MODEL_NAME)

    # Load tokenized data
    logger.info("Loading tokenized data from DuckDB...")
    df = load_tokenized_data(statement)

    # Create embeddings
    logger.info("Creating transaction-level embeddings...")
    create_transaction_embeddings(model, df, "tokenized", txn_collection)

    logger.info("Creating token-level embeddings...")
    create_token_embeddings(model, df, "tokenized", token_collection)

    logger.info("✅ All embeddings created successfully")


if __name__ == "__main__":
    # Initialize logger with script name
    script_name = os.path.basename(__file__).replace('.py', '')
    logger = LoggerFactory.get_logger(get_log_file_path(script_name), config.LOG_LEVEL)
    set_logger(logger)  # Share logger with utils.py

    main()
