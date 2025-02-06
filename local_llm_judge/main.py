import argparse
import logging
import os
import functools
import inspect

import pandas as pd
from prompt_toolkit.history import FileHistory

import local_llm_judge.eval_agent as eval_agent
from local_llm_judge.log_stdout import enable
from local_llm_judge.daydream_data import pairwise_df
from local_llm_judge.image_fetch import fetch_and_resize

logger = logging.getLogger(__name__)


DEFAULT_SEED = 42


def has_inference_uri_arg(fn):
    args = inspect.getfullargspec(fn).args
    return 'inference_uri' in args


def parse_args():
    # List all functions in eval_agent
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-fn', type=str, default='unanimous_ensemble_name_desc')
    parser.add_argument('--N', type=int, default=1000)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--inference-uri', type=str, default='http://localhost:8012/vectorize')
    parser.add_argument('--destroy-cache', action='store_true', default=False)
    parser.add_argument('--check-both-ways', action='store_true', default=False)
    parser.add_argument('--simplify-query', action='store_true', default=False)
    parser.add_argument('--llm-histroy', type=str, default=".llm_shell_history")
    args = parser.parse_args()
    all_fns = eval_agent.all_fns()
    # Funcs to string
    all_fns = [fn.__name__ for fn in all_fns]
    if args.eval_fn not in all_fns:
        logger.info(f"Invalid function name. Available functions: {all_fns}")
        exit(1)
    args.eval_fn = eval_agent.__dict__[args.eval_fn]
    args.cache_key = args.eval_fn.__name__

    if has_inference_uri_arg(args.eval_fn):
        args.eval_fn = functools.partial(args.eval_fn, inference_uri=args.inference_uri)

    if args.check_both_ways:
        cache_key = "both_ways_" + args.eval_fn.__name__
        args.eval_fn = functools.partial(eval_agent.check_both_ways, eval_fn=args.eval_fn)
        args.cache_key = cache_key

    if args.simplify_query:
        args.eval_fn = functools.partial(eval_agent.with_simpler_query, eval_fn=args.eval_fn)
        args.cache_key = "simplify_query_" + args.cache_key

    if args.seed != 42:
        args.cache_key += f"_seed_{args.seed}"

    if args.verbose:
        enable("local_llm_judge")
        enable(__name__)
    return args


def product_row_to_dict(row):
    if 'product_name_x' in row:
        return {
            'id': row['option_id_x'],
            'brand_name': row['brand_name_x'],
            'name': row['product_name_x'],
            'main_image': row['main_image_x'],
            'main_image_path': fetch_and_resize(url=row['main_image_x'], option_id=row['option_id_x']),
            'image_embedding': row['image_embedding_x'],
            'description': row['product_description_x'],
            'category': row['category_x'],
            'grade': row['grade_x']
        }
    elif 'product_name_y' in row:
        return {
            'id': row['option_id_y'],
            'brand_name': row['brand_name_y'],
            'main_image': row['main_image_y'],
            'main_image_path': fetch_and_resize(url=row['main_image_y'], option_id=row['option_id_y']),
            'image_embedding': row['image_embedding_y'],
            'name': row['product_name_y'],
            'description': row['product_description_y'],
            'category': row['category_y'],
            'grade': row['grade_y']
        }


def output_row(query, product_lhs, product_rhs,
               positive_lhs, positive_rhs,
               human_preference, agent_preference):
    return {
        'query': query,
        'product_name_lhs': product_lhs['name'],
        'brand_name_lhs': product_lhs['brand_name'],
        'positive_lhs': positive_lhs,
        'product_description_lhs': product_lhs['description'],
        'option_id_lhs': product_lhs['id'],
        'category_lhs': product_lhs['category'],
        'main_image_lhs': product_lhs['main_image'],
        'main_image_path_lhs': product_lhs['main_image_path'],
        'grade_lhs': product_lhs['grade'],
        'product_name_rhs': product_rhs['name'],
        'brand_name_rhs': product_rhs['brand_name'],
        'product_description_rhs': product_rhs['description'],
        'option_id_rhs': product_rhs['id'],
        'category_rhs': product_rhs['category'],
        'positive_rhs': positive_rhs,
        'main_image_rhs': product_rhs['main_image'],
        'main_image_path_rhs': product_rhs['main_image_path'],
        'grade_rhs': product_rhs['grade'],
        'human_preference': human_preference,
        'agent_preference': agent_preference
    }


def human_pref(query, product_lhs, product_rhs):
    human_preference = product_lhs['grade'] - product_rhs['grade']
    logger.debug(f"Grade LHS: {product_lhs['grade']}, Grade RHS: {product_rhs['grade']}")
    if human_preference > 0:
        return 'LHS'
    elif human_preference < 0:
        return 'RHS'
    else:
        return 'Neither'


def results_df_stats(results_df):
    agent_has_preference = len(results_df[results_df['agent_preference'] != 'Neither']) if (len(results_df) > 0) else 0
    same_preference = len(results_df[results_df['human_preference'] == results_df['agent_preference']]) if (
            len(results_df) > 0) else 0
    no_preference = len(results_df[results_df['agent_preference'] == 'Neither']) if (len(results_df) > 0) else 0
    different_preference = len(results_df[(results_df['human_preference'] != results_df['agent_preference']) & (
            results_df['human_preference'] != 'Neither') & (results_df['agent_preference'] != 'Neither')]) if (
            len(results_df) > 0) else 0
    logger.info(f"Same Preference: {same_preference}," +
                f" Different Preference: {different_preference}, No Preference: {no_preference}")
    if (same_preference + different_preference) > 0:
        precision = same_preference / (same_preference + different_preference) * 100
        recall = agent_has_preference / len(results_df) * 100
        logger.info(f"Precision: {precision:.2f}% | Recall: {recall:.2f}% (N={len(results_df)})")


def has_been_labeled(results_df, query, product_lhs, product_rhs):
    result_exists = (len(results_df) > 0
                     and (results_df[(results_df['query'] == query) &
                          (results_df['option_id_lhs'] == product_lhs['id']) &
                          (results_df['option_id_rhs'] == product_rhs['id'])].shape[0] > 0))
    return result_exists


def main(eval_fn=eval_agent.unanimous_ensemble_name_desc,
         N=1000,
         destroy_cache=False,
         history_path=".llm_shell_history",
         seed=42):
    if history_path:
        eval_agent.qwen.history = FileHistory(history_path)

    df = pairwise_df(N, seed)
    cache_key = args.cache_key
    results_df = pd.DataFrame()
    if destroy_cache and os.path.exists(f'data/features/{cache_key}.pkl'):
        os.remove(f'data/{cache_key}.pkl')
    try:
        results_df = pd.read_pickle(f'data/features/{cache_key}.pkl')
        if 'positive_lhs' not in results_df.columns:
            results_df['positive_lhs'] = True
        if 'positive_rhs' not in results_df.columns:
            results_df['positive_rhs'] = True
        if not isinstance(results_df, pd.DataFrame):
            raise TypeError
    except FileNotFoundError:
        pass
    except TypeError:
        logger.warn("Invalid type of file, going to overwrite")
        results_df = pd.DataFrame()
        pass

    row_num = 0
    for idx, row in df.iterrows():
        query = " ".join(row['user_messages_x'])
        positive_lhs = row['positive_x']
        positive_rhs = row['positive_y']
        product_lhs = product_row_to_dict(row[['product_name_x', 'product_description_x',
                                               'brand_name_x', 'main_image_x',
                                               'image_embedding_x',
                                               'option_id_x', 'category_x', 'grade_x']])
        product_rhs = product_row_to_dict(row[['product_name_y', 'product_description_y',
                                               'brand_name_y', 'main_image_y',
                                               'image_embedding_y',
                                               'option_id_y', 'category_y', 'grade_y']])
        if has_been_labeled(results_df, query, product_lhs, product_rhs):
            logger.info(f"Already rated query: {query}, " +
                        f"product_lhs: {product_lhs['name']}, product_rhs: {product_rhs['name']}")
            logger.info("Skipping")
            continue
        human_preference = human_pref(query, product_lhs, product_rhs)
        logger.info(f"CALLING {cache_key} LLM for query: {query}, " +
                    f"product_lhs({positive_lhs}): {product_lhs['name']}, product_rhs({positive_rhs}): {product_rhs['name']}")
        agent_preference = eval_fn(query, product_lhs, product_rhs)
        if agent_preference != 'Neither' and human_preference != agent_preference:
            logger.warning(f"Disagreement - Human Preference: {human_preference}, Agent Preference: {agent_preference}")
        logger.info(f"Human Preference: {human_preference}, Agent Preference: {agent_preference}")

        results_df = pd.concat([results_df, pd.DataFrame([output_row(query, product_lhs, product_rhs,
                                                                     positive_lhs, positive_rhs,
                                                                     human_preference,
                                                                     agent_preference)])])
        results_df_stats(results_df)

        if row_num % 10 == 0:
            results_df.to_pickle(f'data/features/{cache_key}.pkl')
        row_num += 1
    results_df.to_pickle(f'data/features/{cache_key}.pkl')
    results_df_stats(results_df)


if __name__ == '__main__':
    args = parse_args()
    main(args.eval_fn, args.N, args.destroy_cache, seed=args.seed)
