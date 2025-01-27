import pickle
import argparse
import os
import pandas as pd
import json
import functools
from google.cloud import secretmanager
from google.protobuf import wrappers_pb2
from liaison_client.generated_types.dahlialabs.liaison.v1beta1 import product_pb2

from local_llm_judge import eval_agent
from local_llm_judge.main import product_row_to_dict
from local_llm_judge.train import preference_to_label
from local_llm_judge.es_metadata import es_metadata, es_stag, es_prod
from local_llm_judge.image_fetch import fetch_and_resize

from liaison_client import Client, Config
from liaison_client.helpers import replay_chat
from liaison_client.client import product_pb2
from liaison_client.exceptions import RpcError

secret_manager_client = secretmanager.SecretManagerServiceClient()

def liaison_key(project="dahlia-infra-stag"):
    key = secret_manager_client.access_secret_version(
        name=f"projects/{project}/secrets/liaison-global-apiKey/versions/latest"
    ).payload.data.decode("utf-8")
    return key


@functools.lru_cache()
def stag_liaison():
    uri = 'liaison.stag.dahlialabs.dev:443'
    token = liaison_key(project="dahlia-infra-stag")
    liaison_config = Config(
        host=uri,
        api_token=token,
        default_timeout_ms=30000,
        use_tls=True
    )
    return Client(config=liaison_config)


@functools.lru_cache()
def prod_liaison():
    uri = 'liaison.dahlialabs.dev:443'
    token = liaison_key(project="dahlia-infra-prod")
    liaison_config = Config(
        host=uri,
        api_token=token,
        default_timeout_ms=30000,
        use_tls=True
    )
    return Client(config=liaison_config)


def search_settings_proto_to_dict(search_settings_proto):
    def _to_list(proto_list):
        return [str(item) for item in proto_list]

    def _filter_to_dict(filters):
        return {
            'brand': _to_list(filters.brand),
            'color': _to_list(filters.color),
            'department': _to_list(filters.department),
            'material': _to_list(filters.material),
            'tag': _to_list(filters.tag),
            'category': _to_list(filters.category),
            'option_id': _to_list(filters.option_id),
        }

    search_settings = {
        "search_string": search_settings_proto.search_string,
        "minimum_price": search_settings_proto.minimum_price,
        "maximum_price": search_settings_proto.maximum_price,
        "lexical_search": search_settings_proto.lexical_search,
        "image_search": search_settings_proto.image_search,
    }
    positive_filters = search_settings_proto.positive_filters
    if positive_filters:
        search_settings["positive_filters"] = _filter_to_dict(positive_filters)
    negative_filters = search_settings_proto.negative_filters
    if negative_filters:
        search_settings["negative_filters"] = _filter_to_dict(negative_filters)
    return search_settings


def products_for_msgs(liaison_client, es_client, user_messages, user_id="46467832-d7e9-43b9-9946-e0c00a1f7a76"):
    agent_resp = replay_chat(liaison_client, user_messages, user_id)
    chat_id = agent_resp.id
    listProductsReq = product_pb2.ListProductCardsRequest(
        page_size=10,
        chat_search_context=product_pb2.ListProductCardsSearchContext(
            chat_id=wrappers_pb2.StringValue(value=chat_id)  # pylint: disable=no-member
        )
    )
    listProductsReq.chat_search_context.search_settings.MergeFrom(agent_resp.search_settings)
    search_settings_dict = search_settings_proto_to_dict(agent_resp.search_settings)
    result = liaison_client.product.list_product_cards(request=listProductsReq)
    ranked_options = []
    for product in result.products:
        product_name = product.name
        brand_name = product.brand_data.name
        description = product.description
        for option in product.options:
            if not option.provenance.source:
                break
            main_image_url = None
            for image in option.images:
                if image.is_main:
                    main_image_url = image.src
                    break
            ranked_options.append({
                "product_name": product_name,
                "brand_name": brand_name,
                "option_id": option.id,
                "main_image": main_image_url,
                'main_image_path': fetch_and_resize(main_image_url, option.id),
                "product_description": description,
            })
    df = pd.DataFrame(ranked_options)
    extra_metadata = es_metadata(es_client, df['option_id'].tolist())
    df = df.merge(extra_metadata[['option_id', 'category']], on='option_id')
    df.rename(columns={'option_id': 'id', 'product_name': 'name', 'product_description': 'description'}, inplace=True)
    return df, search_settings_dict


class FeatureCache:

    def __init__(self):
        self.path = "data/feature_cache.pkl"
        try:
            self.cache = pd.read_pickle(self.path)
        except FileNotFoundError:
            self.cache = {}
        print(f"Loaded {len(self.cache)} cached option pair evals")

    def compute_feature(self, feature_fn, feature_name, query, option_lhs, option_rhs):
        key = (feature_name, query, option_lhs['id'], option_rhs['id'])
        if key in self.cache:
            return self.cache[key]
        feature = feature_fn(query, option_lhs, option_rhs)
        self.cache[key] = feature
        with open(self.path, 'wb') as f:
            pickle.dump(self.cache, f)
        return feature


def get_feature_fn(feature_name):
    both_ways = False
    if feature_name.startswith('both_ways_'):
        feature_name = feature_name.replace('both_ways_', '')
        both_ways = True

    eval_fn = eval_agent.__dict__[feature_name]
    if both_ways:
        eval_fn = functools.partial(eval_agent.check_both_ways, eval_fn=eval_fn)
    return eval_fn


def compare_results(model_path, query, results_lhs, results_rhs, cache, thresh=0.8):
    model = None
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    feature_names = model.feature_names_in_
    feature_fns = [get_feature_fn(feature_name) for feature_name in feature_names]

    # Build a dataframe of each feature
    features = []
    for (_, option_lhs), (_, option_rhs) in zip(results_lhs.iterrows(), results_rhs.iterrows()):
        row = {}
        for feature_name, feature_fn in zip(feature_names, feature_fns):
            feature = cache.compute_feature(feature_fn, feature_name, query, option_lhs, option_rhs)
            row[feature_name] = preference_to_label(feature)
        features.append(row)

    # Hack for now give all the features to the non-empty side
    # Which will almost certainly let that side win
    if len(results_lhs) > len(results_rhs):
        for i in range(len(results_lhs) - len(results_rhs)):
            features.append({feature_name: -1 for feature_name in feature_names})
    elif len(results_rhs) > len(results_lhs):
        for i in range(len(results_rhs) - len(results_lhs)):
            features.append({feature_name: 1 for feature_name in feature_names})

    feature_df = pd.DataFrame(features)
    probas = model.predict_proba(feature_df)
    result_rows = []
    for posn in range(len(probas)):
        features = feature_df.iloc[posn]
        result = {
            'query': query,
            'name_lhs': results_lhs.iloc[posn]['name'] if posn < len(results_lhs) else None,
            'option_id_lhs': results_lhs.iloc[posn]['id'] if posn < len(results_lhs) else None,
            'image_url_lhs': results_lhs.iloc[posn]['main_image'] if posn < len(results_lhs) else None,
            'prob_lhs': probas[posn][0],
            'name_rhs': results_rhs.iloc[posn]['name'] if posn < len(results_rhs) else None,
            'option_id_rhs': results_rhs.iloc[posn]['id'] if posn < len(results_rhs) else None,
            'image_url_rhs': results_rhs.iloc[posn]['main_image'] if posn < len(results_rhs) else None,
            'prob_rhs': probas[posn][1],
        }
        for feature_name, feature in features.items():
            if feature == 0:
                result[feature_name] = 'Neither'
            elif feature == -1:
                result[feature_name] = 'LHS'
            elif feature == 1:
                result[feature_name] = 'RHS'
        result_rows.append(result)
    return pd.DataFrame(result_rows)


def get_results(query, dept="w"):
    stag_user_id = "46467832-d7e9-43b9-9946-e0c00a1f7a76"
    prod_user_id = "9c61d6a6-a5fa-4cb4-bde5-485fc3231ae3"
    if dept == "m":
        stag_user_id = "c9d43820-3c20-415a-8a43-059fe0385716"
        prod_user_id = "aa58aa2a-6902-4c86-afdf-62fafdb4b9ac"

    products_stag, search_settings_stag = products_for_msgs(stag_liaison(), es_stag, [query],
                                                            stag_user_id)
    products_prod, search_settings_prod = products_for_msgs(prod_liaison(), es_prod, [query],
                                                            prod_user_id)
    return products_stag, products_prod, search_settings_stag, search_settings_prod


def stag_vs_prod(queries, results_path="data/rated_queries.pkl"):

    cache = FeatureCache()
    model = "data/both_ways_desc_both_ways_category_both_ways_captions_both_ways_brand_both_ways_all_fields.pkl"
    print(f"Comparing {len(queries)} queries")
    result_dfs = []
    for query in queries:
        dept = 'w'
        # If tuple
        if isinstance(query, tuple):
            query, dept = query
        print(f"Processing Query: {query} - Department: {dept}")
        stag, prod, ss_stag, ss_prod = get_results(query, dept)
        df = compare_results(model, query, stag, prod, cache)
        df['ss_lhs'] = json.dumps(ss_stag)
        df['ss_rhs'] = json.dumps(ss_prod)
        result_dfs.append(df)
    results = pd.concat(result_dfs)
    results.to_pickle(results_path)
    return results
