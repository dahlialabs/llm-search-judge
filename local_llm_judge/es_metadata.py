from elasticsearch import Elasticsearch
import os
import pandas as pd


def get_options(es, option_ids):
    resp = es.mget(index="option", body={"ids": option_ids})
    sources = [doc['_source'] for doc in resp.body['docs']
               if doc['found']]
    ids = [doc['_id'] for doc in resp.body['docs']
           if doc['found']]
    for source, id in zip(sources, ids):
        source['option_id'] = id
    return sources


def _es_stag():
    username = os.getenv("ELASTICSEARCH_STAG_USER")
    password = os.getenv("ELASTICSEARCH_STAG_PASSWORD")
    host = "http://localhost:9201"
    es = Elasticsearch(host, basic_auth=(username, password))
    return es


def _es_prod():
    username = os.getenv("ELASTICSEARCH_PROD_USER")
    password = os.getenv("ELASTICSEARCH_PROD_PASSWORD")
    host = "http://localhost:9202"
    es = Elasticsearch(host, basic_auth=(username, password))
    return es


def es_local():
    es = Elasticsearch("http://localhost:9200")
    return es


def es_metadata(es, option_ids, batch_size=100):
    options = []
    for i in range(0, len(option_ids), batch_size):
        batch = option_ids[i:i + batch_size]
        options = options + get_options(es, batch)
    options_df = pd.DataFrame(options)
    options_df = options_df.rename(columns={"name": "product_name"})
    # Dedup on option id
    options_df = options_df.drop_duplicates(subset="option_id")
    return options_df


es_stag = _es_stag()
es_prod = _es_prod()
