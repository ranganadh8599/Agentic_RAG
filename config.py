# Agentic RAG
# Central configuration, loaded from environment variables (.env file).

import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    # --- Database (PostgreSQL) ---
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/agentic_rag"
    )

    # --- Embeddings (any litellm-supported model) ---
    # Examples:
    #   "openai/text-embedding-3-small"     (1536 dims, needs OPENAI_API_KEY)
    #   "gemini/text-embedding-004"         (768 dims, needs GEMINI_API_KEY)
    #   "ollama/nomic-embed-text"           (768 dims, local via Ollama)
    #   "mock"                              (no key needed, for offline testing)
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "mock")
    EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", "1536"))

    # Asymmetric query/document prefixes: a big retrieval-quality win for
    # sentence-transformers models (BAAI/bge, nomic, all-MiniLM). Usually OFF
    # for API embeddings (OpenAI/Gemini) — keep OFF for those, ON for local/Ollama.
    USE_ASYMMETRIC_PREFIX: bool = os.getenv("USE_ASYMMETRIC_PREFIX", "0") == "1"
    QUERY_PREFIX: str = "Represent the query for retrieving relevant documents: "
    DOC_PREFIX: str = "Represent the document for retrieval: "

    # --- LLM (litellm model strings, ANY provider) ---
    # Examples: "openai/gpt-4o-mini", "gemini/gemini-2.0-flash",
    #           "anthropic/claude-3-5-sonnet", "ollama/llama3"
    #   "mock" = no key needed, for offline testing.
    LLM_MODEL: str = os.getenv("LLM_MODEL", "mock")
    VISION_MODEL: str = os.getenv("VISION_MODEL", "mock")
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.2"))
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "1024"))

    # --- Chunking (LibreChat-proven defaults: 1500/100) ---
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))

    # Short structured documents (resumes, forms, profiles) get even LARGER chunks so
    # their sections stay intact and aren't fragmented mid-sentence.
    STRUCTURED_CHUNK_SIZE: int = int(os.getenv("STRUCTURED_CHUNK_SIZE", "2500"))
    STRUCTURED_CHUNK_OVERLAP: int = int(os.getenv("STRUCTURED_CHUNK_OVERLAP", "200"))
    STRUCTURED_MAX_CHARS: int = int(os.getenv("STRUCTURED_MAX_CHARS", "5000"))

    # --- Retrieval (borrowed from Onyx + LixSearch) ---
    TOP_K: int = int(os.getenv("TOP_K", "6"))
    RELEVANCE_FLOOR: float = float(os.getenv("RELEVANCE_FLOOR", "0.25"))
    RRF_K: int = int(os.getenv("RRF_K", "60"))          # reciprocal-rank fusion constant
    USE_QUERY_EXPANSION: bool = os.getenv("USE_QUERY_EXPANSION", "1") == "1"
    SEMANTIC_CACHE_THRESHOLD: float = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.90"))
    # A query is treated as "really about a document" only when its top vector
    # match is at least this cosine (gemini-embedding-2: strong ~0.75+, weak ~0.6).
    # Below it, general-knowledge questions fall back to the general model.
    GENERAL_STRONG_THRESHOLD: float = float(os.getenv("GENERAL_STRONG_THRESHOLD", "0.72"))

    # --- Ingestion ---
    MAX_IMAGES_PER_PDF: int = int(os.getenv("MAX_IMAGES_PER_PDF", "20"))
    MAX_IMAGE_DIM: int = int(os.getenv("MAX_IMAGE_DIM", "1200"))
    MAX_PAGES: int = int(os.getenv("MAX_PAGES", "0"))  # 0 = whole document
    # Filenames that signal a short structured doc (resume/form) -> larger chunks.
    # Comma-separated, case-insensitive substring match.
    STRUCTURED_KEYWORDS: str = os.getenv(
        "STRUCTURED_KEYWORDS", "resume,cv,curriculum,profile,form,application")
    # How many texts to embed+store per batch (incremental saving).
    EMBED_BATCH_SIZE: int = int(os.getenv("EMBED_BATCH_SIZE", "32"))

    # --- Uploads (size / count) ---
    # Max size of a single uploaded file in MB (0 = unlimited). Enforced
    # server-side (413) and pre-checked in the UI.
    MAX_UPLOAD_MB: int = int(os.getenv("MAX_UPLOAD_MB", "0"))
    # Max files accepted per upload batch/selection (0 = unlimited). Enforced
    # in the UI (extra files are skipped with a notice).
    MAX_UPLOAD_FILES: int = int(os.getenv("MAX_UPLOAD_FILES", "0"))

    # --- Agents / citations ---
    MAX_CRITIC_ROUNDS: int = int(os.getenv("MAX_CRITIC_ROUNDS", "2"))
    ROUTER_MAX_TOKENS: int = int(os.getenv("ROUTER_MAX_TOKENS", "100"))
    CRITIC_MAX_TOKENS: int = int(os.getenv("CRITIC_MAX_TOKENS", "300"))
    # A "padding" citation is pruned when its chunk shares less than this
    # fraction of the claim's significant words.
    CITATION_OVERLAP_THRESHOLD: float = float(os.getenv("CITATION_OVERLAP_THRESHOLD", "0.25"))
    # Source-card snippets: max characters and the overlap-search window (words).
    SNIPPET_MAX_CHARS: int = int(os.getenv("SNIPPET_MAX_CHARS", "220"))
    SNIPPET_WINDOW: int = int(os.getenv("SNIPPET_WINDOW", "40"))

    # --- Retrieval knobs ---
    # Size of the query-embedding LRU cache (repeated queries never re-embed).
    QUERY_EMBED_CACHE_SIZE: int = int(os.getenv("QUERY_EMBED_CACHE_SIZE", "256"))
    # Keyword search: multiplier applied to a document-title ts_rank match.
    KEYWORD_TITLE_BOOST: float = float(os.getenv("KEYWORD_TITLE_BOOST", "2.0"))
    # Score used for exact file-name matches (promoted to the top of results).
    FILENAME_MATCH_SCORE: float = float(os.getenv("FILENAME_MATCH_SCORE", "3.0"))
    # Fetch N x top_k candidates per search before fusion.
    RETRIEVAL_MULTIPLIER: int = int(os.getenv("RETRIEVAL_MULTIPLIER", "2"))

    # --- Conversation memory ---
    MEMORY_RECENT_K: int = int(os.getenv("MEMORY_RECENT_K", "8"))
    MEMORY_RELEVANT_K: int = int(os.getenv("MEMORY_RELEVANT_K", "4"))

    # --- Vision / images ---
    IMAGE_JPEG_QUALITY: int = int(os.getenv("IMAGE_JPEG_QUALITY", "85"))
    VISION_SUMMARY_MAX_TOKENS: int = int(os.getenv("VISION_SUMMARY_MAX_TOKENS", "500"))

    # --- pgvector ---
    # The `vector` type's HNSW index supports at most this many dimensions
    # (pgvector platform limit). Above it we store embeddings as `halfvec`
    # (HNSW cap 4000) so fast indexing still works.
    HNSW_VECTOR_DIM_LIMIT: int = int(os.getenv("HNSW_VECTOR_DIM_LIMIT", "2000"))

    # --- MongoDB (users, sessions & chat history — LibreChat-style) ---
    # RAG data (documents/chunks/collections/cache) stays in PostgreSQL/pgvector.
    MONGO_URI: str = os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017")
    MONGO_DB: str = os.getenv("MONGO_DB", "agentic_rag")
    SESSION_TTL_SECONDS: int = int(os.getenv("SESSION_TTL_SECONDS", "2592000"))  # 30 days


settings = Settings()
