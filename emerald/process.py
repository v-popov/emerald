import re
import spacy
import string
import numpy as np
import pandas as pd
from tdda.rexpy import rexpy
from itertools import product
from fuzzywuzzy import fuzz
from fuzzywuzzy import process as fuzz_process
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from pandas.api.types import is_string_dtype, is_numeric_dtype


NLP = spacy.load('en_core_web_sm')


# Get every combination for character n-grams for lowercase letters & numbers
NGRAM_RANGE = (3, 3)
VOCABULARY = []
for i in range(NGRAM_RANGE[0], NGRAM_RANGE[1]+1):
    for token in product(string.ascii_lowercase+string.digits+string.whitespace,
                         repeat=i):
        VOCABULARY.append(''.join(token))


def remove_special_characters(text):
    return re.sub(r"[^a-zA-Z0-9]+", ' ', text)


def get_entity_type(values):
    entities = []
    for value in values:
        doc = NLP(value)
        entities.extend([X.label_ for X in doc.ents])
    if entities:
        primary_entity = max(set(entities), key=entities.count)
        if primary_entity in ["GPE", "LOC", "FAC"]:
            return "address"
        elif primary_entity == "PERSON":
            return "name"
    else:
        return None


def contains_digits(values, max_prop=0.5):
    prob_sum = 0
    for value in values:
        proportion = sum(c.isdigit() for c in str(value)) / len(str(value))
        if proportion >= max_prop:
            prob_sum += 1
    return prob_sum / len(values)


def contains_name_or_address(values, column_name):
    # Calculate probability from values
    entity_count = 0
    for value in values:
        entity_present = False
        doc = NLP(value)
        for X in doc.ents:
            if X.label_ in ["GPE", "LOC", "FAC", "ORG", "PERSON"]:
                entity_present = True
        if entity_present:
            entity_count += 1
    values_prob = entity_count / len(values)
    # Calculate probability from column name
    valid_names = ["address", "city", "state", "street", "country", "name"]
    _, cname_prob = fuzz_process.extractOne(
        remove_special_characters(column_name), valid_names, scorer=fuzz.partial_ratio
    )
    cname_prob = cname_prob / 100
    # Return combined probability
    return values_prob*0.4  + cname_prob*0.6


def compute_text_vector(values, vocabulary):
    # Compute n-gram range from the vocabulary
    ngram_range = (min(len(token) for token in vocabulary),
                   max(len(token) for token in vocabulary))
    # Concatenate into single string and strip punctuation
    input_string = ' '.join(values)
    input_string = input_string.translate(str.maketrans('', '', string.punctuation))
    vectorizer = CountVectorizer(lowercase=True, analyzer='char', ngram_range=ngram_range,
                                 vocabulary=vocabulary, binary=True, dtype=np.float32)
    vector = vectorizer.fit_transform([input_string])
    return vector


def compute_cosine_similarity(vector_a, vector_b, mask_zeroes=False):
    if mask_zeroes:
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


def process_table(table, max_sample_size=10):
    processed_columns = []
    for column_name in table:
        column_metadata = {}
        column_metadata["column_name"] = column_name

        # Sample unique values from the column
        sample_values = table[column_name].value_counts().index[:max_sample_size]

        # Process the data based on its data classification
        if is_numeric_dtype(table[column_name]):
            column_metadata["data_class"] = "numeric_or_id"
            column_metadata["regex"] = rexpy.extract(sample_values.astype('str'))[0]

        elif is_string_dtype(table[column_name]):

            if contains_digits(sample_values, max_prop=0.5) > 0.8:
                column_metadata["data_class"] = "numeric_or_id"
                column_metadata["regex"] = rexpy.extract(sample_values.astype('str'))[0]

            elif contains_name_or_address(sample_values, column_name) > 0.65:
                column_metadata["data_class"] = "name_or_address"
                column_metadata["vector"] = compute_text_vector(sample_values, VOCABULARY)
            else:
                column_metadata["data_class"] = "text"
                column_metadata["vector"] = compute_text_vector(sample_values, VOCABULARY)

        processed_columns.append(column_metadata)
    return processed_columns
