IDEA: Should one store differnt chunking stores???? for diferen strategies..

    Hybrid search — The AI now runs vector similarity + keyword text search in parallel, then merges and deduplicates results. Exact matches (like model_slug, def model_slug) are found via ILIKE and prioritized with [EXACT] tags.

    Smart keyword extraction — Automatically detects code identifiers (snake_case, camelCase, quoted terms) from your question and runs targeted text searches for them.

    Smaller chunks — Reduced default chunk size from 1500→800 chars with max chunks per file increased from 20→40, so individual functions/classes get their own chunks instead of being buried in large blocks.

Action needed: Re-sync your repos (Codebase → Sync) to re-chunk with the smaller size. The hybrid search works immediately for keyword matching on existing chunks.