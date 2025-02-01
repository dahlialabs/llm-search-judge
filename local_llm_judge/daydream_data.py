import pandas as pd
import numpy as np

from local_llm_judge.image_fetch import fetch_and_resize

import logging

logger = logging.getLogger(__name__)


def _build_pairwise_df(labeled_df, n, seed=42):
    # Map grades 1,2 to 1
    labeled_df['grade'] = labeled_df['grade'].map({1: 1, 2: 1, 3: 2, 4: 2, 5: 3})
    # Drop the 2s
    labeled_df = labeled_df[labeled_df['grade'] != 2]
    labeled_df = labeled_df[~labeled_df['product_name'].isnull()]
    # Get pairwise
    pairwise = labeled_df.merge(labeled_df, on='query_id')
    # Shuffle completely, otherwise they're somewhat sorted on query
    pairwise = pairwise.sample(frac=1, random_state=seed)

    # Drop same id
    pairwise = pairwise[pairwise['option_id_x'] != pairwise['option_id_y']]

    # Drop same rating
    pairwise = pairwise[pairwise['grade_x'] != pairwise['grade_y']]

    assert n <= len(pairwise), f"Only {len(pairwise)} rows available"
    logger.info(f"Pairs available: {len(pairwise)}")
    return pairwise.head(n)


img_root = "https://cdn.stag.dahlialabs.dev/fotomancer/_/rs:fit:512:1024/plain/gs://dahlia-stag-gcs-public-assets/"


def _img_proxy_url(img_url):
    img_proxy = img_root + img_url
    return img_proxy


def pairwise_df(n, seed=42, sample_negatives=True):
    labeled_df = pd.read_parquet('data/labeled_options.parquet')
    logger.info(f"Loaded {len(labeled_df)} labeled options")
    # labeled_df['main_image'] = labeled_df['main_image'].apply(_img_proxy_url)
    labeled_df.rename(columns={'description': 'product_description', 'rating': 'grade'}, inplace=True)
    for option_id in labeled_df[~labeled_df['main_image'].isna()]['option_id'].unique():
        image_url = labeled_df[labeled_df['option_id'] == option_id]['main_image'].iloc[0]
        fetch_and_resize(image_url, option_id)
    labeled_df.drop_duplicates(subset=['query_id', 'option_id'], inplace=True)
    labeled_df['positive'] = True

    # The user messages of this id look like the agent is speaking, not the user
    bad_query_ids = ['3dbe47dd-09f1-425a-a430-a7e53d376f5d']
    labeled_df = labeled_df[~labeled_df['query_id'].isin(bad_query_ids)]

    # Fix a trick question
    trick_question = '696de9a0-49ac-4808-96cf-1f33837e5abb'

    def remove_trick(x):
        if x['query_id'] == trick_question:
            x['user_messages'][0] = ""
            x['user_messages'][1] = x['user_messages'][1].replace("nevermind.", "I")
        return x
    labeled_df = labeled_df.apply(remove_trick, axis=1)
    labeled_df['user_messages_concat'] = labeled_df['user_messages'].apply(lambda x: " ".join(x))

    if sample_negatives:
        logger.info("Sampling negatives")
        # Others positives as my negatives to give obvious negative cases
        good_results = labeled_df[labeled_df['grade'] == 5].set_index('query_id')
        neg_labels = []
        for query_id in labeled_df['query_id'].unique():
            if query_id not in good_results.index:
                continue
            dest_query = labeled_df[labeled_df['query_id'] == query_id].copy()
            other_query_results = good_results[good_results.index != query_id].copy().reset_index()
            other_query_results = other_query_results.sample(frac=1, random_state=seed)

            other_query_results = other_query_results[~other_query_results['query_id'].isna()]
            other_query_results.loc[:, 'query_id'] = other_query_results['query_id'].astype(str)
            other_query_results.loc[:, 'grade'] = np.int64(1)
            other_query_results.loc[:, 'query_id'] = str(query_id)
            # other_query_results.loc[: 'user_messages'] = dest_query['user_messages'].iloc[0]
            # ASsign list of user_messages with tile
            my_msgs = other_query_results.apply(lambda x: dest_query['user_messages'].iloc[0], axis=1)
            other_query_results['user_messages'] = my_msgs

            logger.debug(f"New Negative {other_query_results.iloc[0]}")

            neg_labels.append(other_query_results)

        neg_labels_df = pd.concat(neg_labels).sample(frac=1, random_state=seed)
        neg_labels_df = neg_labels_df[~neg_labels_df['main_image'].isna()]
        neg_labels_df = neg_labels_df.head(400)
        neg_labels_df['positive'] = False

        # Get some example negatives to log
        logger.info(f"Example negative {neg_labels_df.iloc[10]}")

        labeled_df = pd.concat([labeled_df, neg_labels_df])

    pairs = _build_pairwise_df(labeled_df, n, seed)
    logger.info(f"Returning {len(pairs)} pairs")
    return pairs
