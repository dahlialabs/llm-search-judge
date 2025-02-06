import pickle
import pandas as pd
import json
import functools
import logging
import os
import inspect

from local_llm_judge import eval_agent
from local_llm_judge.train import preference_to_label
# from local_llm_judge.search_backend import stag, prod, local, stag_cached, prod_cached, local_es
from local_llm_judge.image_fetch import fetch_and_resize


log = logging.getLogger(__name__)


class FeatureCache:
    """Remember the LLM evals we've already computed"""

    def __init__(self, overwrite=False):
        self.overwrite = overwrite
        self.path = "~/.local-llm-judge/feature_cache.pkl"
        self.path = os.path.expanduser(self.path)
        try:
            self.cache = pd.read_pickle(self.path)
        except FileNotFoundError:
            self.cache = {}
        log.info(f"Loaded {len(self.cache)} cached option pair evals")

    def compute_feature(self, feature_fn, feature_name, query, option_lhs, option_rhs):
        key = (feature_name, query, option_lhs['id'], option_rhs['id'])
        if key in self.cache and not self.overwrite:
            return self.cache[key]
        log.info(f"Computing uncached feature: {feature_name} for {query} - {option_lhs['id']} vs {option_rhs['id']}")
        feature = feature_fn(query, option_lhs, option_rhs)
        self.cache[key] = feature
        with open(self.path, 'wb') as f:
            pickle.dump(self.cache, f)
        return feature


def get_feature_fn(feature_name, inference_uri="http://localhost:8012/vectorize"):
    both_ways = False
    simplify_query = False
    if feature_name.startswith('both_ways_'):
        feature_name = feature_name.replace('both_ways_', '')
        both_ways = True

    if feature_name.startswith('simplify_query_'):
        feature_name = feature_name.replace('simplify_query_', '')
        simplify_query = True

    eval_fn = eval_agent.__dict__[feature_name]
    # Bind inference_uri
    params = inspect.signature(eval_fn).parameters
    if 'inference_uri' in params:
        eval_fn = functools.partial(eval_fn, inference_uri=inference_uri)
    if both_ways:
        eval_fn = functools.partial(eval_agent.check_both_ways, eval_fn=eval_fn)
    if simplify_query:
        eval_fn = functools.partial(eval_agent.with_simpler_query, eval_fn=eval_fn)
    return eval_fn


def _build_feature_df(cache, feature_names, query, results_lhs, results_rhs,
                      inference_uri="http://localhost:8012/vectorize"):
    # Build a dataframe of each feature
    feature_fns = [get_feature_fn(feature_name, inference_uri) for feature_name in feature_names]
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
    return feature_df


def compare_results(model_path, query, results_lhs, results_rhs, cache, thresh=0.8,
                    inference_uri="http://localhost:8012/vectorize"):
    model = None
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    feature_names = model.feature_names_in_
    log.info(f"Comparing {len(results_lhs)} vs {len(results_rhs)}")
    log.info(f"Features: {feature_names}")
    feature_df = _build_feature_df(cache, feature_names, query, results_lhs, results_rhs,
                                   inference_uri=inference_uri)

    # Predict
    probas = model.predict_proba(feature_df)
    result_rows = []

    # Report on output
    for posn in range(len(probas)):
        features = feature_df.iloc[posn]
        result = {
            'query': query,
            'name_lhs': results_lhs.iloc[posn]['name'] if posn < len(results_lhs) else None,
            'brand_name_lhs': results_lhs.iloc[posn]['brand_name'] if posn < len(results_lhs) else None,
            'category_lhs': results_lhs.iloc[posn]['category'] if posn < len(results_lhs) else None,
            'desc_lhs': results_lhs.iloc[posn]['description'] if posn < len(results_lhs) else None,
            'backend_lhs': results_lhs.iloc[posn]['backend'] if posn < len(results_lhs) else None,
            'image_embedding_lhs': results_lhs.iloc[posn]['image_embedding'] if posn < len(results_lhs) else None,
            'option_id_lhs': results_lhs.iloc[posn]['id'] if posn < len(results_lhs) else None,
            'image_url_lhs': results_lhs.iloc[posn]['main_image'] if posn < len(results_lhs) else None,
            'pref_lhs': probas[posn][0],
            'name_rhs': results_rhs.iloc[posn]['name'] if posn < len(results_rhs) else None,
            'brand_name_rhs': results_rhs.iloc[posn]['brand_name'] if posn < len(results_rhs) else None,
            'desc_rhs': results_rhs.iloc[posn]['description'] if posn < len(results_rhs) else None,
            'category_rhs': results_rhs.iloc[posn]['category'] if posn < len(results_rhs) else None,
            'image_embedding_rhs': results_rhs.iloc[posn]['image_embedding'] if posn < len(results_rhs) else None,
            'backend_rhs': results_rhs.iloc[posn]['backend'] if posn < len(results_rhs) else None,
            'option_id_rhs': results_rhs.iloc[posn]['id'] if posn < len(results_rhs) else None,
            'image_url_rhs': results_rhs.iloc[posn]['main_image'] if posn < len(results_rhs) else None,
            'pref_rhs': probas[posn][1],
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


def fetch_images(search_results):
    for _, row in search_results.iterrows():
        if row['main_image'] is None:
            continue
        if row['main_image_path'] is not None:
            continue
        row['main_image_path'] = fetch_and_resize(row['main_image'], row['option_id'])


def fetch_results(query, dept, lhsEnv, rhsEnv):
    products_lhs, search_settings_lhs = lhsEnv.search(query, dept)
    fetch_images(products_lhs)
    if len(products_lhs) == 0:
        log.warn(f"No results for query: {query} - Department: {dept} from {lhsEnv.name}")

    products_rhs, search_settings_rhs = rhsEnv.search(query, dept)
    fetch_images(products_rhs)
    if len(products_rhs) == 0:
        log.warn(f"No results for query: {query} - Department: {dept} from {rhsEnv.name}")

    return products_lhs, products_rhs, search_settings_lhs, search_settings_rhs


def compare_env(model_path, queries, backend_lhs, backend_rhs,
                overwrite_feature_cache=False,
                inference_uri="http://localhost:8012/vectorize"):
    cache = FeatureCache(overwrite=overwrite_feature_cache)
    # model = "data/both_ways_desc_both_ways_category_both_ways_captions_both_ways_brand_both_ways_all_fields.pkl"
    log.info(f"Comparing {len(queries)} queries")
    result_dfs = []
    for query in queries:
        dept = 'w'
        # If tuple
        if isinstance(query, tuple):
            query, dept = query
        log.info(f"Processing Query: {query} - Department: {dept}")
        stag_results, prod_results, ss_stag, ss_prod = fetch_results(query, dept, backend_lhs, backend_rhs)
        df = compare_results(model_path, query, stag_results, prod_results, cache, inference_uri=inference_uri)
        df['ss_lhs'] = json.dumps(ss_stag)
        df['ss_rhs'] = json.dumps(ss_prod)
        result_dfs.append(df)
    results = pd.concat(result_dfs)
    return results
