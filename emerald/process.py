import spacy
import string
import numpy as np
import pandas as pd
from tdda.rexpy import rexpy
from itertools import product
from fuzzywuzzy import fuzz
from fuzzywuzzy import process as fuzz_process
from sklearn.feature_extraction.text import CountVectorizer
from pandas.api.types import is_string_dtype, is_numeric_dtype

from . import utils

NLP = spacy.load("en_core_web_sm")


# Get every combination for character n-grams for lowercase letters & numbers
NGRAM_RANGE = (3, 3)
VOCABULARY = []
for i in range(NGRAM_RANGE[0], NGRAM_RANGE[1]+1):
    for token in product(string.ascii_lowercase+string.digits+string.whitespace,
                         repeat=i):
        VOCABULARY.append(''.join(token))


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
    valid_names = ["addr", "address", "city", "state", "street", "country", "name"]
    _, cname_prob = fuzz_process.extractOne(
        utils.remove_special_characters(column_name), valid_names, scorer=fuzz.token_set_ratio
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


def process(table, max_sample_size=10, identifiers=None):
    processed_columns = []
    for column_name in table:
        column_metadata = {}
        if identifiers:
            for key, value in identifiers.items():
                column_metadata[key] = value
        column_metadata["column_name"] = column_name

        # Sample unique values from the column
        sample_values = table[column_name].value_counts().index[:max_sample_size]
        # sample_values = np.array([x for x in sample_values if str(x) != 'nan'], dtype=str)

        # Process the data based on its data classification
        if is_numeric_dtype(table[column_name]):
            column_metadata["data_class"] = "numeric_or_id"
            column_metadata["representation"] = rexpy.extract(sample_values.astype('str'))[0]

        elif is_string_dtype(table[column_name]):

            if contains_digits(sample_values, max_prop=0.5) > 0.8:
                column_metadata["data_class"] = "numeric_or_id"
                column_metadata["representation"] = rexpy.extract(sample_values.astype('str'))[0]

            elif contains_name_or_address(sample_values, column_name) > 0.65:
                column_metadata["data_class"] = "named_entity"
                column_metadata["representation"] = compute_text_vector(sample_values, VOCABULARY)
            else:
                column_metadata["data_class"] = "text"
                column_metadata["representation"] = compute_text_vector(sample_values, VOCABULARY)

        processed_columns.append(column_metadata)
    return processed_columns
