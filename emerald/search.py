import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz
from sklearn.metrics.pairwise import cosine_similarity

from . import utils


def regex_similarity(regex_a, regex_b):
    return 1.0


def vector_similarity(vector_a, vector_b, partial=False):
    if partial:
        if len(vector_a.nonzero()[1]) <= len(vector_b.nonzero()[1]):
            nonzero_indices = vector_a.nonzero()[1]
        else:
            nonzero_indices = vector_b.nonzero()[1]
        return cosine_similarity(vector_a[0, nonzero_indices],
                                 vector_b[0, nonzero_indices],
                                 dense_output=True)[0][0]
    else:
        return cosine_similarity(vector_a,
                                 vector_b,
                                 dense_output=True)[0][0]


def column_similarity(column_a, column_b):
    name_similarity = fuzz.token_set_ratio(
        utils.remove_special_characters(column_a["column_name"]),
        utils.remove_special_characters(column_b["column_name"])
    ) / 100.0
    if column_a["data_class"] == "numeric_or_id" and column_b["data_class"] == "numeric_or_id":
        data_similarity = regex_similarity(
            column_a["representation"], column_b["representation"]
        )
        return (name_similarity * 0.7) + (data_similarity * 0.3)
    elif column_a["data_class"] == "named_entity" and column_b["data_class"] == "named_entity":
        data_similarity = vector_similarity(
            column_a["representation"], column_b["representation"], partial=True
        )
        return (name_similarity * 0.7) + (data_similarity * 0.3)
    elif column_a["data_class"] == "text" and column_b["data_class"] == "text":
        data_similarity = vector_similarity(
            column_a["representation"], column_b["representation"], partial=True
        )
        return (name_similarity * 0.4) + (data_similarity * 0.6)
    elif ((column_a["data_class"] == "text" and column_b["data_class"] == "named_entity") or
          (column_a["data_class"] == "named_entity" and column_b["data_class"] == "text")):
        data_similarity = vector_similarity(
            column_a["representation"], column_b["representation"], partial=True
        )
        return (name_similarity * 0.7) + (data_similarity * 0.3)
    else:
        return 0.0


def search(query, data, threshold):
    # Find the column(s) whose name matches the query
    query_columns = data[data['column_name'] == query]
    query_columns = query_columns.reset_index(drop=True)
    if query_columns.empty:
        return []
    else:
        # Compute similarity of the query columns with the known data columns
        similarity_scores = np.zeros((len(data), len(query_columns)))
        for i, query_column in query_columns.iterrows():
            for j, data_column in data.iterrows():
                similarity_scores[j, i] = column_similarity(query_column,
                                                            data_column)
        # Get matched columns with similarity scores above the threshold
        max_scores = np.max(similarity_scores, axis=1)
        matches = data.iloc[max_scores > threshold].copy()
        matches["score"] = np.round(max_scores[max_scores > threshold], 2)
        matches = matches.drop(columns=["data_class", "representation"])
        return matches.to_dict("records")
