import pickle
import argparse
import os
import pandas as pd
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
    return df


def get_feature_fn(feature_name):
    both_ways = False
    if feature_name.startswith('both_ways_'):
        feature_name = feature_name.replace('both_ways_', '')
        both_ways = True

    eval_fn = eval_agent.__dict__[feature_name]
    if both_ways:
        eval_fn = functools.partial(eval_agent.check_both_ways, eval_fn=eval_fn)
    return eval_fn


def compare_results(model_path, query, results_lhs, results_rhs, thresh=0.8):
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
            feature = feature_fn(query, option_lhs, option_rhs)
            row[feature_name] = preference_to_label(feature)
        features.append(row)
    feature_df = pd.DataFrame(features)
    probas = model.predict_proba(feature_df)
    result_rows = []
    for posn in range(len(probas)):
        result = {
            'query': query,
            'name_lhs': results_lhs.iloc[posn]['name'],
            'option_id_lhs': results_lhs.iloc[posn]['id'],
            'image_url_lhs': results_lhs.iloc[posn]['main_image'],
            'prob_lhs': probas[posn][0],
            'name_rhs': results_rhs.iloc[posn]['name'],
            'option_id_rhs': results_rhs.iloc[posn]['id'],
            'image_url_rhs': results_rhs.iloc[posn]['main_image'],
            'prob_rhs': probas[posn][1],
        }
        result_rows.append(result)
    return pd.DataFrame(result_rows)


def get_results(query):
    stag_user_id = "46467832-d7e9-43b9-9946-e0c00a1f7a76"
    products_stag = products_for_msgs(stag_liaison(), es_stag, [query],
                                      stag_user_id)
    prod_user_id = "9c61d6a6-a5fa-4cb4-bde5-485fc3231ae3"
    products_prod = products_for_msgs(prod_liaison(), es_prod, [query],
                                      prod_user_id)
    return products_stag, products_prod


def stag_vs_prod(queries, results_path="data/rated_queries.pkl"):
    model = "data/both_ways_desc_both_ways_category_both_ways_captions_both_ways_brand_both_ways_all_fields.pkl"
    result_dfs = []
    try:
        result_dfs = [pd.read_pickle(results_path)]
        print(f"Loaded {len(result_dfs[0])} results")
        existing_queries = result_dfs[0]['query'].unique()
        print(f"Skipping {len(existing_queries)} queries - {existing_queries}")
        queries = [query for query in queries if query not in result_dfs[0]['query'].unique()]
    except FileNotFoundError:
        result_dfs = []
    print(f"Comparing {len(queries)} queries")
    for query in queries:
        stag, prod = get_results(query)
        df = compare_results(model, query, stag, prod)
        result_dfs.append(df)
    results = pd.concat(result_dfs)
    results.to_pickle(results_path)
    return results


if __name__ == "__main__":
    stag_vs_prod()
