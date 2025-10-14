import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

def semantic_search(filtered_df: pd.DataFrame, user_query: str, top_k: int = None) -> pd.DataFrame:
    """Perform semantic search on the filtered DataFrame.
    
    The method concatenates several processed text columns and computes the cosine similarity
    between the user query and each property. If the DataFrame is empty or no text columns
    are available, it returns a head slice.
    """
    if filtered_df.empty:
        logger.warning("Cannot perform semantic search on empty dataset")
        return filtered_df

    # If top_k is not specified, use all properties
    if top_k is None:
        top_k = len(filtered_df)
    else:
        top_k = min(top_k, len(filtered_df))

    # Define the text columns to use for search.
    search_columns = []
    for field in ['Processed_Location', 'Processed_City', 'Processed_Country', 'Processed_Type', 'Processed_Description', 'Title']:
        if field in filtered_df.columns:
            search_columns.append(field)
    
    if not search_columns:
        logger.warning("No text columns available for semantic search")
        return filtered_df.head(top_k)
    
    # Create a concatenated text for each row
    search_texts = filtered_df.apply(
        lambda row: ' '.join(str(row.get(col, '')) for col in search_columns),
        axis=1
    )
    
    try:
        # Append query text (lowercased) to the corpus for vectorization
        corpus = list(search_texts) + [user_query.lower()]
        tfidf_matrix = TfidfVectorizer(stop_words='english').fit_transform(corpus)
        similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]
        top_indices = similarities.argsort()[-top_k:][::-1]
        return filtered_df.iloc[top_indices]
    except Exception as e:
        logger.error(f"Semantic search error: {e}")
        return filtered_df.head(top_k)