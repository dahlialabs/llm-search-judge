import pandas as pd


def _build_pairwise_df(labeled_df, n, seed=42):
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


def pairwise_df(n, seed=42):
    labeled_df = pd.read_parquet('data/labeled_options.parquet')
    labeled_df.rename(columns={'description': 'product_description', 'rating': 'grade'}, inplace=True)
    return _build_pairwise_df(labeled_df, n, seed)
