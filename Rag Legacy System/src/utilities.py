# Chunking, helpers for reranking and diversity

import logging
import numpy as np

logger = logging.getLogger(__name__)

#LLM-BASED CHUNK SCORING 

def rerank_chunks_scored(question, docs, client, model_id):
    
    scored_docs = []

    for doc in docs:
        try:
            text = doc.page_content[:1800]  

            prompt = f"""
You are an AI judge. Rate how relevant the following document chunk is
for answering this query on a scale of 0 to 3.

Query:
{question}

Chunk:
{text}

Return ONLY a number between 0 and 3.
"""

            response = client.converse(
                modelId=model_id,
                messages=[
                    {
                        "role": "user",
                        "content": [{"text": prompt}]
                    }
                ],
                inferenceConfig={"maxTokens": 20, "temperature": 0.0}
            )

            score_text = response["output"]["message"]["content"][0]["text"].strip()
            score = int(score_text) if score_text.isdigit() else 0

            scored_docs.append((doc, score))

        except Exception as e:
            logger.error(f"[RERANKING] Error scoring doc: {e}")
            scored_docs.append((doc, 0))

    scored_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)
    return scored_docs



#PAGE-BASED SOURCE DIVERSITY

def apply_page_diversity(scored_docs):
    seen = set()
    diversified = []

    for doc, score in scored_docs:
        src = doc.metadata.get("source")
        page = doc.metadata.get("page")
        key = (src, page)

        if key not in seen:
            seen.add(key)
            diversified.append((doc, score))

    return diversified


#COSINE SIMILARITY

def cosine_similarity(vec1, vec2):
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)

    if denom == 0:
        return 0.0

    return float(np.dot(v1, v2) / denom)


#MAXIMUM MARGINAL RELEVANCE (MMR)

def apply_mmr(scored_docs, embeddings, top_k=5, lambda_param=0.7):
    
    if not scored_docs:
        return []

    selected = []
    used_indices = set()

    doc_embs = np.array([embeddings[i] for i in range(len(scored_docs))])

    selected.append(scored_docs[0])
    used_indices.add(0)

    while len(selected) < min(top_k, len(scored_docs)):
        best = None
        best_score = -999
        best_idx = None

        for i in range(len(scored_docs)):
            if i in used_indices:
                continue

            relevance = scored_docs[i][1]
            diversity_penalty = max(
                cosine_similarity(doc_embs[i], doc_embs[j]) for j in used_indices
            )

            mmr_value = lambda_param * relevance - (1 - lambda_param) * diversity_penalty

            if mmr_value > best_score:
                best_score = mmr_value
                best = scored_docs[i]
                best_idx = i

        selected.append(best)
        used_indices.add(best_idx)

    return selected
