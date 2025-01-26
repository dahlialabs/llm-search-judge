import pandas as pd


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
    return pairwise.head(n)


img_root = "https://cdn.stag.dahlialabs.dev/fotomancer/_/rs:fit:512:1024/plain/gs://dahlia-stag-gcs-public-assets/"


def _img_proxy_url(img_url):
    img_proxy = img_root + img_url
    return img_proxy


def pairwise_df(n, seed=42):
    labeled_df = pd.read_parquet('data/labeled_options.parquet')
    # labeled_df['main_image'] = labeled_df['main_image'].apply(_img_proxy_url)
    labeled_df.rename(columns={'description': 'product_description', 'rating': 'grade'}, inplace=True)
    labeled_df.drop_duplicates(subset=['query_id', 'option_id'], inplace=True)
    return _build_pairwise_df(labeled_df, n, seed)
