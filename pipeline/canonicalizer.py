"""
Supervised Token Canonicalization Pipeline (Incremental + Versioned)
 -------------------------------------------------------------------
- Learns token→canonical mappings from labeled transaction data.
- Incrementally updates existing mappings with new labeled samples.
- Automatically writes versioned local artifacts.
- LLM-based entity extraction for identifying entity + bank/platform
"""

import os
import re
import json
import numpy as np
import pandas as pd
from functools import lru_cache
from . import config
from .utils import *
from datetime import datetime
from collections import defaultdict, Counter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz, process
import duckdb
import warnings
warnings.filterwarnings('ignore')

from pydantic import BaseModel, Field

logger = get_logger()


# ============================================================
# 1. LLM Entity Extraction Schema
# ============================================================

class EntityExtraction(BaseModel):
    """Schema for extracting entity and metadata from tokenized transaction."""
    entity: Optional[str] = Field(default="", description="The main entity/merchant name (e.g., Flipkart, Starbucks)")
    bank: Optional[str] = Field(default="", description="Bank name if present in tokens")
    platform: Optional[str] = Field(default="", description="Platform/service name if present (e.g., CAMS, Paytm)")
    transaction_type: Optional[str] = Field(default="", description="Type: shopping, investment, bill_payment, transfer, etc.")


def create_entity_extractor():
    """Initialize the LLM for entity extraction."""
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        base_url=config.LLM_CONFIG.LOCAL_LLM_BASE_URL,
        api_key=config.LLM_CONFIG.LOCAL_LLM_API_KEY,
        model=config.LLM_CONFIG.LOCAL_LLM_MODEL,
        temperature=config.LLM_CONFIG.TEMPERATURE,
        max_tokens=128,
        timeout=30,
        max_retries=0,
    )
    return llm.with_structured_output(EntityExtraction)


def extract_entity_llm(tokens: list) -> EntityExtraction:
    """Extract entity and metadata from tokenized list using LLM."""
    try:
        extractor = create_entity_extractor()
        prompt = f"""Analyze the following transaction tokens and extract the entity and associated metadata.

Rules:
- entity: The main merchant/person name (the one doing the transaction)
- bank: Bank names like hdfc, sbi, icici, kotak, axis, yesbank, indusind, bandhan, etc.
- platform: Payment platforms like paytm, phonepe, gpay, razorpay, cams, etc.
- transaction_type: shopping, investment, bill_payment, transfer, food, transport, utilities, other

Examples:
Input: ['flipkart', 'yesbank']
Output: {{"entity": "flipkart", "bank": "yesbank", "platform": "", "transaction_type": "shopping"}}

Input: ['ppfas', 'cams']
Output: {{"entity": "ppfas", "bank": "", "platform": "cams", "transaction_type": "investment"}}

Input: ['vishwendr', 'hdfcbank']
Output: {{"entity": "vishwendr", "bank": "hdfcbank", "platform": "", "transaction_type": "transfer"}}

Now analyze:
Input: {tokens}"""

        result = extractor.invoke(prompt)

        # Clean up None values
        if result.entity is None:
            result.entity = ""
        if result.bank is None:
            result.bank = ""
        if result.platform is None:
            result.platform = ""
        if result.transaction_type is None:
            result.transaction_type = ""

        return result
    except Exception as e:
        print(f"⚠️ LLM extraction failed for {tokens}: {e}")
        return EntityExtraction(entity="", bank="", platform="", transaction_type="")


def extract_entities_batch(token_lists: list, batch_size: int = 15) -> list:
    """Extract entities in small batches; identical token lists are sent once."""
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        base_url=config.LLM_CONFIG.LOCAL_LLM_BASE_URL,
        api_key=config.LLM_CONFIG.LOCAL_LLM_API_KEY,
        model=config.LLM_CONFIG.LOCAL_LLM_MODEL,
        temperature=config.LLM_CONFIG.TEMPERATURE,
        max_tokens=600,
        timeout=180,
        max_retries=0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

    unique = list(dict.fromkeys(tuple(t) for t in token_lists))
    by_tokens = {}
    total_batches = (len(unique) + batch_size - 1) // batch_size
    print(f"🔍 {len(token_lists)} rows -> {len(unique)} unique token sets, {total_batches} batches")

    from tqdm import tqdm
    pbar = tqdm(range(0, len(unique), batch_size), desc="LLM extraction", unit="batch")
    total_parsed = 0
    for b, i in enumerate(pbar, start=1):
        batch = unique[i:i + batch_size]
        prompt = (
            "For each numbered transaction output exactly one line: "
            "idx|entity|bank|platform|transaction_type\n"
            "entity = main merchant/person; bank = hdfc, sbi, icici, axis, yesbank etc. or empty; "
            "platform = paytm, phonepe, gpay, razorpay, cams etc. or empty; "
            "transaction_type = shopping, investment, bill_payment, transfer, food, transport, utilities, other.\n"
            "Example: 1|flipkart|yesbank||shopping\n"
            "No other text.\n\nTransactions:"
        )
        for n, tokens in enumerate(batch, start=1):
            prompt += f"\n{n}. {list(tokens)}"

        try:
            text = llm.invoke(prompt).content or ""
            parsed = 0
            for line in text.splitlines():
                parts = [p.strip() for p in line.split("|")]
                if len(parts) < 5 or not parts[0].isdigit():
                    continue
                idx = int(parts[0])
                if 1 <= idx <= len(batch):
                    by_tokens[batch[idx - 1]] = EntityExtraction(
                        entity=parts[1], bank=parts[2],
                        platform=parts[3], transaction_type=parts[4],
                    )
                    parsed += 1
            total_parsed += parsed
            pbar.set_postfix(parsed=f"{total_parsed}/{min(i + batch_size, len(unique))}")
        except Exception as e:
            tqdm.write(f"  ⚠️ Batch {b}/{total_batches} failed: {e}")

    empty = EntityExtraction(entity="", bank="", platform="", transaction_type="")
    results = [by_tokens.get(tuple(t), empty).model_copy() for t in token_lists]

    # Clean up None values (LLM can return null instead of "")
    for r in results:
        if r.entity is None:
            r.entity = ""
        if r.bank is None:
            r.bank = ""
        if r.platform is None:
            r.platform = ""
        if r.transaction_type is None:
            r.transaction_type = ""

    return results


# ============================================================
# 1b. Helper Functions
# ============================================================

def clean_mapping_keys(mapping: dict) -> dict:
    """Clean mapping keys - remove quotes and normalize."""
    cleaned = {}
    for key, value in mapping.items():
        # Remove surrounding quotes from keys
        clean_key = str(key).strip().strip("'\"")
        cleaned[clean_key] = value
    return cleaned


# ============================================================
# 2. Load Data
# ============================================================

def load_training_data(limit=10000):
    """Loads labels joined to the pipeline's own tokens, so training and inference share one vocabulary."""
    conn = create_duckdb_connection(config.DATABASE_PATH)
    clean = lambda col: f"regexp_replace(lower({col}), '[^a-z0-9]', '', 'g')"
    # Manual ids use another hash and its text has no spaces, so match on date + amount + cleaned text.
    df = conn.query(f"""
        WITH j AS (
            SELECT t.tokenized, m.normalized_trans_src,
                   row_number() OVER (PARTITION BY m.trans_id ORDER BY t.trans_id) AS rn
            FROM {config.SCHEMA_NAME}.acc_labelled m
            JOIN {config.SCHEMA_NAME}.{config.tables('account').tokens} t
              ON CAST(m.trans_dt AS DATE) = CAST(t.trans_dt AS DATE)
                         AND TRY_CAST(REPLACE(CAST(m.trans_amt AS VARCHAR), ',', '') AS DOUBLE)
                                 = TRY_CAST(t.trans_amt AS DOUBLE)
             AND {clean('m.particulars')} = {clean('t.particulars')}
            WHERE m.normalized_trans_src IS NOT NULL
        )
        SELECT tokenized, normalized_trans_src FROM j WHERE rn = 1
        LIMIT {limit}
    """).to_df()

    print(f"✅ Loaded {len(df)} labeled records")
    return df

def load_new_data(statement="account", limit=10000):
    """Loads the tokenized transactions of one statement type from DuckDB."""
    conn = create_duckdb_connection(config.DATABASE_PATH)
    df = conn.query(f"""
        SELECT *
        FROM {config.SCHEMA_NAME}.{config.tables(statement).tokens}
    """).to_df()

    print(f"✅ Loaded {len(df)} new records")
    return df

# ============================================================
# 3. Token Extraction
# ============================================================

def extract_tokens(value):
    """Convert tokenized field into clean lowercase tokens."""
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return [str(t).strip().lower() for t in value if t]
    elif isinstance(value, str):
        parts = re.findall(r"[A-Za-z0-9'.]+", value.lower())
        return [p.strip() for p in parts if p]
    return []

# ============================================================
# 4. Token → Canonical Mapping
# ============================================================

def compute_token_entropy(train_df):
    """Compute Shannon entropy for each token across sources."""
    from scipy.stats import entropy
    import numpy as np

    token_src_counts = defaultdict(Counter)
    for _, row in train_df.iterrows():
        for tok in row["tokens"]:
            tok = str(tok).strip().strip("'\"")
            token_src_counts[tok][row["normalized_trans_src"]] += 1

    token_entropy = {}
    for token, counts in token_src_counts.items():
        total = sum(counts.values())
        probs = [c / total for c in counts.values()]
        token_entropy[token] = entropy(probs, base=2)

    return token_entropy


def build_token_mapping(train_df, existing_map=None,
                       entropy_threshold=config.ENTROPY_THRESHOLD,
                       dominance_threshold=config.DOMINANCE_THRESHOLD,
                       min_count=config.MIN_TOKEN_COUNT):
    """Build token→canonical mappings using IDF + statistical gates."""
    import math
    from scipy.stats import entropy as compute_entropy

    # Step 1: compute IDF across all tokens
    # High IDF = rare across transactions = likely an entity name
    token_doc_freq = defaultdict(int)
    all_rows = []

    for _, row in train_df.iterrows():
        tokens = [str(t).strip().strip("'\"").lower() for t in row["tokens"]]
        all_rows.append((tokens, row["normalized_trans_src"]))
        for tok in set(tokens):  # set → count doc freq, not term freq
            token_doc_freq[tok] += 1

    N = len(train_df) if len(train_df) > 0 else 1
    token_idf = {tok: math.log(N / (1 + df)) for tok, df in token_doc_freq.items()}

    # Step 2: learn from every token; entropy/dominance gates below drop ambiguous ones
    token_src_counts = defaultdict(Counter)
    for tokens, src in all_rows:
        for tok in set(tokens):
            if tok:
                token_src_counts[tok][src] += 1

    # Step 3: apply entropy + dominance gates as before
    token_entropy = {
        tok: compute_entropy([c / sum(counts.values()) for c in counts.values()])
        for tok, counts in token_src_counts.items()
    }

    token_to_canonical = dict(existing_map or {})
    stats = defaultdict(int)

    for token, counts in token_src_counts.items():
        total = sum(counts.values())
        top_src, top_count = counts.most_common(1)[0]

        if total < min_count:
            stats["rare"] += 1; continue
        if token_entropy.get(token, 0) > entropy_threshold:
            stats["entropy"] += 1; continue
        if top_count / total < dominance_threshold:
            stats["dominance"] += 1; continue

        token_to_canonical[token] = top_src
        stats["kept"] += 1

    print(f"📊 kept={stats['kept']} | rare={stats['rare']} "
          f"| entropy={stats['entropy']} | dominance={stats['dominance']}")
    return token_to_canonical

# ============================================================
# 5. Embedding Functions
# ============================================================

def build_source_embeddings(model, trans_src_list):
    """Encodes canonical source names for semantic similarity."""
    src_embeddings = model.encode(trans_src_list, convert_to_numpy=True, normalize_embeddings=True)
    return np.array(src_embeddings), trans_src_list

# ============================================================
# 6. Normalization Functions
# ============================================================

# ============================================================
# 6b. Fuzzy Token Matching (for typos/variations)
# ============================================================

# Fuzzy match threshold for token matching
FUZZY_TOKEN_THRESHOLD = 80

def find_fuzzy_token_match(token: str, token_to_canonical: dict, threshold: int = FUZZY_TOKEN_THRESHOLD) -> str:
    """Find fuzzy match for token in the mapping."""
    if not token:
        return None

    token = str(token).lower().strip()
    # Same first letter only: stops "others" matching the learned token "mother"
    mapping_tokens = [t for t in token_to_canonical if t[:1] == token[:1]]
    if not mapping_tokens:
        return None

    match = process.extractOne(token, mapping_tokens, scorer=fuzz.ratio)
    if match:
        matched_token, score, _ = match
        if score >= threshold:
            return token_to_canonical[matched_token]

    return None


BANK_TOKENS = {'hdfc', 'sbi', 'icici', 'kotak', 'axis', 'yesbank', 'indusind', 'bandhan',
               'punjab', 'union', 'canara', 'federal', 'idfc', 'citi', 'hsbc',
               'yesb', 'utib', 'sbin', 'hdfcb', 'neft', 'rtgs', 'imps', 'upi',
               'ach', 'nach', 'card', 'pos', 'atm', 'bank', 'cc', 'debit', 'credit'}


@lru_cache(maxsize=1)
def _load_cross_encoder():
    """Load the local cross-encoder only when candidate rankings are ambiguous."""
    try:
        from sentence_transformers import CrossEncoder

        return CrossEncoder(config.CROSS_ENCODER_MODEL)
    except Exception as exc:
        logger.warning("Cross-encoder unavailable; leaving ambiguous source unmatched: %s", exc)
        return None


def _reciprocal_rank_fusion(rankings, limit=10, rank_constant=60):
    """Fuse ranked candidate lists without treating their scores as comparable."""
    fused = defaultdict(float)
    for ranking in rankings:
        for rank, candidate in enumerate(dict.fromkeys(ranking), start=1):
            fused[candidate] += 1.0 / (rank_constant + rank)
    return sorted(fused, key=fused.get, reverse=True)[:limit], fused


def _compact_name(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _lexical_source_rank(tokens, token_to_canonical, src_names):
    """Rank sources from direct/fuzzy token evidence and name-token overlap."""
    counts = Counter()
    for token in tokens:
        source = token_to_canonical.get(token)
        if source is None:
            source = find_fuzzy_token_match(token, token_to_canonical)
        if source in src_names:
            counts[source] += 1
    return sorted(
        counts,
        key=lambda source: (counts[source] + _name_coverage(source, tokens), counts[source]),
        reverse=True,
    )


def _semantic_source_rank(tokens, model, src_names, src_embeddings, limit=10):
    if not tokens or src_embeddings is None or not src_names:
        return [], None
    vector = model.encode(
        [" ".join(tokens)],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    similarities = cosine_similarity(vector, src_embeddings)[0]
    order = np.argsort(similarities)[::-1][:limit]
    return [src_names[int(index)] for index in order], similarities


def _cross_encoder_pick(tokens, context, candidates, cross_encoder):
    if len(candidates) < 2:
        return candidates[0] if candidates else None
    cross_encoder = cross_encoder or _load_cross_encoder()
    if cross_encoder is None:
        return None

    query = "" if context is None or pd.isna(context) else str(context).strip()
    token_text = " ".join(tokens)
    if query and token_text not in query.lower():
        query = f"{query} Tokens: {token_text}"
    elif not query:
        query = token_text
    pairs = [(f"Transaction: {query}", f"Merchant/source: {name}") for name in candidates]
    try:
        scores = np.asarray(cross_encoder.predict(pairs, show_progress_bar=False)).reshape(-1)
    except Exception as exc:
        logger.warning("Cross-encoder scoring failed; leaving ambiguous source unmatched: %s", exc)
        return None
    if len(scores) != len(candidates):
        logger.warning("Cross-encoder returned %s scores for %s candidates", len(scores), len(candidates))
        return None
    order = np.argsort(scores)[::-1]
    if len(order) > 1 and scores[order[0]] - scores[order[1]] < config.CROSS_ENCODER_MIN_MARGIN:
        return None
    return candidates[int(order[0])]


def normalize_transaction_hybrid(
    tokens,
    model,
    token_to_canonical,
    src_names,
    src_embeddings,
    similarity_threshold=config.SIMILARITY_THRESHOLD,
    context=None,
    cross_encoder=None,
):
    """Fuse lexical and semantic candidates, reranking only ambiguous rows."""
    if not tokens:
        return None

    tokens = [str(t).lower().strip() for t in tokens if t]
    if not tokens:
        return None

    non_bank_tokens = [t for t in tokens if t not in BANK_TOKENS]
    bank_tokens = [t for t in tokens if t in BANK_TOKENS]
    if not src_names:
        return _try_match(non_bank_tokens or bank_tokens, token_to_canonical)

    # A compact multi-token span ("mother" + "hood" -> "motherhood") is
    # stronger evidence than a mapping for one shared token ("mother"). Ignore
    # surrounding location/reference tokens such as "noida" when finding spans.
    phrase = "".join(non_bank_tokens)
    phrase_variants = {phrase} if len(phrase) >= 5 else set()
    for start in range(len(non_bank_tokens)):
        for end in range(start + 2, len(non_bank_tokens) + 1):
            span = "".join(non_bank_tokens[start:end])
            if len(span) >= 5:
                phrase_variants.add(span)
    phrase_candidates = {
        name
        for name in src_names
        if any(variant in _compact_name(name) for variant in phrase_variants)
    }
    if len(phrase_candidates) == 1:
        return next(iter(phrase_candidates))

    lexical_rank = _lexical_source_rank(non_bank_tokens, token_to_canonical, src_names)
    if not lexical_rank and bank_tokens:
        lexical_rank = _lexical_source_rank(bank_tokens, token_to_canonical, src_names)

    semantic_rank, similarities = _semantic_source_rank(
        non_bank_tokens or bank_tokens, model, src_names, src_embeddings
    )
    phrase_rank = []
    if phrase:
        phrase_matches = sorted(
            ((fuzz.ratio(phrase, _compact_name(name)), name) for name in src_names),
            reverse=True,
        )
        phrase_rank = [name for score, name in phrase_matches[:10] if score >= 40]

    rankings = [ranking for ranking in (lexical_rank, semantic_rank, phrase_rank) if ranking]
    candidates, _ = _reciprocal_rank_fusion(rankings, limit=5)
    if not candidates:
        return None

    ranking_tops = {ranking[0] for ranking in rankings if ranking}
    rankings_conflict = len(ranking_tops) > 1
    semantic_uncertain = False
    if similarities is not None and len(similarities) > 1:
        top_two = np.sort(similarities)[-2:]
        semantic_uncertain = top_two[-1] - top_two[-2] < 0.05

    if not lexical_rank and similarities is not None and float(np.max(similarities)) < similarity_threshold:
        return None

    if len(candidates) > 1 and (rankings_conflict or semantic_uncertain):
        return _cross_encoder_pick(tokens, context, candidates, cross_encoder)

    best = candidates[0]
    if similarities is None and not lexical_rank:
        return None
    if similarities is not None and not lexical_rank:
        if similarities[src_names.index(best)] < similarity_threshold:
            return None
    return best


def _name_coverage(src: str, tokens: list) -> int:
    """How many transaction tokens appear inside the source name (e.g. 'motherhoodhospital')."""
    name = re.sub(r"[^a-z0-9]", "", src.lower())
    return sum(1 for t in set(tokens) if len(t) >= 3 and t in name)


def _best_source(matches: list, tokens: list) -> str:
    """Most-voted source; tokens found in the source name break ties and outweigh single stray votes."""
    counts = Counter(matches)
    return max(counts, key=lambda s: counts[s] + _name_coverage(s, tokens))


def _try_match(tokens: list, token_to_canonical: dict) -> str:
    """Try direct + fuzzy matching on given tokens."""
    if not tokens:
        return None

    # Direct mapping
    direct_matches = [token_to_canonical[t] for t in tokens if t in token_to_canonical]
    if direct_matches:
        return _best_source(direct_matches, tokens)

    # Fuzzy matching
    fuzzy_matches = []
    for token in tokens:
        fuzzy_result = find_fuzzy_token_match(token, token_to_canonical)
        if fuzzy_result:
            fuzzy_matches.append(fuzzy_result)

    if fuzzy_matches:
        return _best_source(fuzzy_matches, tokens)

    return None


def llm_classify_unmatched(df, results, src_names, batch_size=10):
    """Let the LLM pick a known source for unmatched rows; replies outside src_names are discarded."""
    from tqdm import tqdm
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        base_url=config.LLM_CONFIG.LOCAL_LLM_BASE_URL,
        api_key=config.LLM_CONFIG.LOCAL_LLM_API_KEY,
        model=config.LLM_CONFIG.LOCAL_LLM_MODEL,
        temperature=config.LLM_CONFIG.TEMPERATURE,
        max_tokens=200,
        timeout=120,
        max_retries=0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

    todo = {}
    for toks, text, res in zip(df["canonical_tokens"], df[config.CORPUS_COL], results):
        if res is None:
            todo.setdefault(tuple(toks), str(text)[:100])
    keys = list(todo)
    print(f"🤖 {sum(r is None for r in results)} unmatched rows -> {len(keys)} unique token sets")

    catalog = "\n".join(f"{i}. {s}" for i, s in enumerate(src_names, start=1))
    answers = {}
    pbar = tqdm(range(0, len(keys), batch_size), desc="LLM source match", unit="batch")
    for i in pbar:
        batch = keys[i:i + batch_size]
        prompt = (
            f"Known sources:\n{catalog}\n\n"
            "For each numbered transaction output one line 'n|k', where k is the number of the "
            "known source it was paid to or received from, or 0 if none clearly matches. "
            "Do not guess. No other text.\n\nTransactions:"
        )
        for n, key in enumerate(batch, start=1):
            prompt += f"\n{n}. {todo[key]} | tokens: {list(key)}"

        try:
            text = llm.invoke(prompt).content or ""
            for line in text.splitlines():
                m = re.match(r"\s*(\d+)\s*\|\s*(\d+)", line)
                if m and 1 <= int(m[1]) <= len(batch) and 1 <= int(m[2]) <= len(src_names):
                    answers[batch[int(m[1]) - 1]] = src_names[int(m[2]) - 1]
        except Exception as e:
            tqdm.write(f"  ⚠️ Batch failed: {e}")
        pbar.set_postfix(matched=len(answers))

    return [r if r is not None else (f"[AI] {answers[tuple(t)]}" if tuple(t) in answers else None)
            for t, r in zip(df["canonical_tokens"], results)]


def normalize_transaction(
    tokens,
    model,
    token_to_canonical,
    src_names,
    src_embeddings,
    similarity_threshold=config.SIMILARITY_THRESHOLD,
    context=None,
    cross_encoder=None,
):
    """Normalize a full transaction."""
    return normalize_transaction_hybrid(
        tokens,
        model,
        token_to_canonical,
        src_names,
        src_embeddings,
        similarity_threshold,
        context=context,
        cross_encoder=cross_encoder,
    )


def normalize_token(token, model, token_to_canonical, src_names, src_embeddings, similarity_threshold=config.SIMILARITY_THRESHOLD):
    """Normalize a single token."""
    if not token:
        return None
    token = str(token).lower().strip()

    if token in token_to_canonical:
        return token_to_canonical[token]

    vec = model.encode([token], convert_to_numpy=True, normalize_embeddings=True)
    sims = cosine_similarity(vec, src_embeddings)[0]
    best_idx = np.argmax(sims)
    best_sim = sims[best_idx]
    return src_names[best_idx] if best_sim >= similarity_threshold else None


def normalize_tokens_batch(tokens: list, model, token_to_canonical, src_names, src_embeddings, similarity_threshold=config.SIMILARITY_THRESHOLD):
    """Normalize a batch of tokens using batch encoding."""
    if not tokens:
        return []

    # Filter tokens that have direct mapping
    results = {}
    tokens_to_encode = []

    for token in tokens:
        token = str(token).lower().strip()
        if not token:
            results[token] = None
        elif token in token_to_canonical:
            results[token] = token_to_canonical[token]
        else:
            tokens_to_encode.append(token)

    # Batch encode remaining tokens
    if tokens_to_encode:
        batch_embeddings = model.encode(tokens_to_encode, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        similarities = cosine_similarity(batch_embeddings, src_embeddings)

        for idx, token in enumerate(tokens_to_encode):
            best_idx = np.argmax(similarities[idx])
            best_sim = similarities[idx][best_idx]
            results[token] = src_names[best_idx] if best_sim >= similarity_threshold else None

    return [results.get(t, None) for t in tokens]


# ============================================================
# 7. Artifact Versioning
# ============================================================

def get_next_version(base_dir=config.ARTIFACT_DIR):
    """Determine next artifact version based on existing folders."""
    os.makedirs(base_dir, exist_ok=True)
    existing = [d for d in os.listdir(base_dir) if re.match(r"v\d+", d)]
    if not existing:
        return "v1"
    nums = [int(d[1:]) for d in existing]
    next_num = max(nums) + 1
    return f"v{next_num}"

def save_versioned_artifacts(version, token_to_canonical, src_names, src_embeddings):
    """Save artifacts under artifacts/<version>/"""
    vdir = os.path.join(config.ARTIFACT_DIR, version)
    os.makedirs(vdir, exist_ok=True)

    token_path = os.path.join(vdir, "token_to_canonical_supervised.json")
    src_names_path = os.path.join(vdir, "src_names.npy")
    src_emb_path = os.path.join(vdir, "src_embeddings.npy")

    with open(token_path, "w", encoding="utf8") as f:
        json.dump(token_to_canonical, f, indent=2, ensure_ascii=False)
    np.save(src_names_path, np.array(src_names, dtype=object), allow_pickle=True)
    np.save(src_emb_path, src_embeddings)

    # Update "latest" symlinks for convenience
    latest_links = {
        os.path.join(config.ARTIFACT_DIR, "latest_token_to_canonical.json"): token_path,
        os.path.join(config.ARTIFACT_DIR, "latest_src_names.npy"): src_names_path,
        os.path.join(config.ARTIFACT_DIR, "latest_src_embeddings.npy"): src_emb_path,
    }
    for link, target in latest_links.items():
        # Force remove both file and symlink if exists
        if os.path.islink(link):
            os.unlink(link)
        elif os.path.exists(link):
            os.remove(link)
        # Create relative symlink
        os.symlink(os.path.relpath(target, config.ARTIFACT_DIR), link)

    print(f"💾 Saved versioned artifacts → {vdir}")
    return vdir

# ============================================================
# 8. Train and Normalize Helper
# ============================================================

def train_token_canonicalizer(incremental=True):
    df = load_training_data()
    df["tokens"] = df["tokenized"].apply(extract_tokens)
    df = df[df["tokens"].map(len) > 0]
    print(f"🧹 Cleaned {len(df)} rows with valid tokens")

    # Load previous "latest" artifacts if available
    existing_map, src_names, src_embeddings = {}, [], None
    if incremental and os.path.exists(os.path.join(config.ARTIFACT_DIR, "latest_token_to_canonical.json")):
        print("♻️ Loading existing mappings for incremental update...")
        raw_map = json.load(open(os.path.join(config.ARTIFACT_DIR, "latest_token_to_canonical.json")))
        existing_map = clean_mapping_keys(raw_map)
        src_names = np.load(os.path.join(config.ARTIFACT_DIR, "latest_src_names.npy"), allow_pickle=True).tolist()
        src_embeddings = np.load(os.path.join(config.ARTIFACT_DIR, "latest_src_embeddings.npy"))

    # Build or update canonical mapping
    token_to_canonical = build_token_mapping(df, existing_map)

    # Update embeddings
    model = SentenceTransformer(config.MODEL_NAME)
    new_src_names = sorted(set(df["normalized_trans_src"].dropna().unique()))
    src_names = sorted(set(src_names) | set(new_src_names))
    src_embeddings, src_names = build_source_embeddings(model, src_names)

    # Save under new version folder
    version = get_next_version()
    vdir = save_versioned_artifacts(version, token_to_canonical, src_names, src_embeddings)

def normalize_new_data(output_to="duckdb", use_llm_entity_extraction=False, statement="account"):
    """Apply the canonicalization model to one statement type's tokenized data."""
    df = load_new_data(statement)
    df["tokens"] = df["tokenized"].apply(extract_tokens)
    df = df[df["tokens"].map(len) > 0]
    print(f"🧹 Cleaned {len(df)} rows with valid tokens")
    if len(df) == 0:
        print("No new data to normalize.")
        return

    # Step 1: LLM-based Entity Extraction (Batched)
    if use_llm_entity_extraction:
        print("🔍 Extracting entities using LLM (batched)...")
        token_lists = df["tokens"].tolist()
        entity_results = extract_entities_batch(token_lists, batch_size=15)
        df["entity"] = [r.entity for r in entity_results]
        df["bank"] = [r.bank for r in entity_results]
        df["platform"] = [r.platform for r in entity_results]
        df["trans_type_llm"] = [r.transaction_type for r in entity_results]

        # Keep the original tokens (the map was trained on them) and add the entity's words
        df["canonical_tokens"] = df.apply(
            lambda row: row["tokens"] + [
                w for w in re.findall(r"[a-z0-9]+", (row["entity"] or "").lower())
                if w not in row["tokens"]
            ],
            axis=1
        )
        print("✅ Entity extraction complete")
    else:
        df["canonical_tokens"] = df["tokens"]

    # Load latest artifacts
    model = SentenceTransformer(config.MODEL_NAME)
    raw_mapping = json.load(open(os.path.join(config.ARTIFACT_DIR, "latest_token_to_canonical.json")))
    token_to_canonical = clean_mapping_keys(raw_mapping)
    src_names = np.load(os.path.join(config.ARTIFACT_DIR, "latest_src_names.npy"), allow_pickle=True).tolist()
    src_embeddings = np.load(os.path.join(config.ARTIFACT_DIR, "latest_src_embeddings.npy"))

    from tqdm import tqdm
    results = []
    pbar = tqdm(
        zip(df["canonical_tokens"], df[config.CORPUS_COL]),
        desc="Normalizing",
        unit="row",
        total=len(df),
    )
    for toks, context in pbar:
        results.append(normalize_transaction(
            toks, model, token_to_canonical, src_names, src_embeddings, context=context
        ))
        pbar.set_postfix(matched=sum(r is not None for r in results))

    if config.LLM_CONFIG.ENABLE_FALLBACK:
        results = llm_classify_unmatched(df, results, src_names)
    df["normalized_trans_src"] = results

    print("✅ Normalization complete.")
    print(df[["tokenized", "normalized_trans_src"]].head(10))

    # Save results
    if output_to == "duckdb":
        conn = create_duckdb_connection(config.DATABASE_PATH)
        conn.register("df_temp", df)
        conn.execute(f"""
            CREATE OR REPLACE TABLE {config.SCHEMA_NAME}.{config.tables(statement).normalized} AS
            SELECT * FROM df_temp
        """)
        print(f"💾 Normalized data saved to DuckDB (table: {config.tables(statement).normalized})")
    else:
        out_path = os.path.join(config.ARTIFACT_DIR, f"normalized_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        df.to_csv(out_path, index=False)
        print(f"💾 Normalized data saved to {out_path}")

# ============================================================
# 8. Main Pipeline
# ============================================================

def run_pipeline(train=False, normalize=False, incremental=True, statement="account"):
    # Mappings are learned from labelled account data only and shared by every statement type
    if train and statement == "account":
        print("=== Training Token Canonicalizer ===")
        train_token_canonicalizer(incremental=incremental)
    if normalize:
        print(f"=== Normalizing {statement} data ===")
        normalize_new_data(use_llm_entity_extraction=False, statement=statement)



# ============================================================
# 9. Run
# ============================================================
if __name__ == "__main__":
    # Initialize logger with script name
    script_name = os.path.basename(__file__).replace('.py', '')
    logger = LoggerFactory.get_logger(get_log_file_path(script_name), config.LOG_LEVEL)
    set_logger(logger)  # Share logger with utils.py

    run_pipeline(
        train=config.CN_TRAIN,
        normalize=config.CN_NORMALIZE,
        incremental=config.CN_INCREMENTAL
        )
