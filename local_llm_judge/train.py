import argparse
import pandas as pd
import numpy as np
import warnings
import os
import logging
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import export_text, export_graphviz
from sklearn.model_selection import KFold

import itertools
import graphviz
import pickle

from local_llm_judge.log_stdout import enable

log = logging.getLogger(__name__)

enable('log_llm_judge')
enable(__name__)


def preference_to_label(preference):
    if preference == 'LHS':
        return -1
    elif preference == 'RHS':
        return 1
    else:
        return 0


def visualize_tree(tree_model, feature_names):
    dot_data = export_graphviz(tree_model, out_file=None,
                               feature_names=feature_names,
                               filled=True, rounded=True,
                               special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.view()


def build_feature_df(feature_names):
    feature_df = None
    feature_columns = []
    for feature in feature_names:
        df = pd.read_pickle(feature)[['query', 'option_id_lhs', 'option_id_rhs',
                                      'agent_preference', 'human_preference']]
        feature_name = os.path.basename(feature).split(".")[0]
        feature_columns.append(feature_name)
        df.rename(columns={"agent_preference": feature_name}, inplace=True)
        if feature_df is None:
            feature_df = df
        else:
            feature_df = feature_df.merge(df, on=['query', 'option_id_lhs', 'option_id_rhs', 'human_preference'],
                                          how='inner')
        feature_df[feature_name] = feature_df[feature_name].apply(preference_to_label)
    feature_df['human_preference'] = feature_df['human_preference'].apply(preference_to_label)
    feature_df = feature_df[~feature_df.index.duplicated(keep='first')]
    return feature_df, feature_columns


def parse_args():
    # Get feature names in argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_names", type=str, nargs="+")
    args = parser.parse_args()
    return args


def train_tree(train, feature_columns):
    clf = DecisionTreeClassifier()
    clf.fit(train[feature_columns],
            train['human_preference'])
    return clf


def predict(clf, test, feature_columns, threshold=0.9):
    """Only assign LHS or RHS if the probability is above the threshold"""
    probas = clf.predict_proba(test[feature_columns])
    definitely_lhs = probas[:, 0] > threshold
    definitely_rhs = probas[:, 1] > threshold
    predictions = np.array([0] * len(test))
    predictions[definitely_lhs] = -1
    predictions[definitely_rhs] = 1

    test.loc[:, 'prediction'] = predictions
    same_label_when_pred = (
        test[test['prediction'] != 0]['human_preference'] == test[test['prediction'] != 0]['prediction']
    )
    if len(same_label_when_pred) == 0:
        precision = 0
        recall = 0
    else:
        precision = same_label_when_pred.sum() / len(same_label_when_pred)
        recall = len(same_label_when_pred) / len(test)
    log.debug(f"P: {precision} - R: {recall}")

    return predictions, feature_columns, precision, recall


def train_gbt(train, feature_columns):
    clf = GradientBoostingClassifier()
    clf.fit(train[feature_columns],
            train['human_preference'])
    return clf


def permute_features(feature_columns):
    """Return a list of all possible permutations of the feature columns"""
    permutations = []
    for i in range(1, len(feature_columns) + 1):
        permutations.extend(itertools.combinations(feature_columns, i))
    return permutations


def kfold_trees(feature_names, feature_df):
    results = []
    for permutation in permute_features(feature_names):
        permutation = list(permutation)
        kf = KFold(n_splits=5)
        precisions = []
        recalls = []
        for train_index, test_index in kf.split(feature_df):
            # Use kf to define test/train splits
            train = feature_df.iloc[train_index]
            test = feature_df.iloc[test_index]
            clf = train_tree(train, permutation)
            model_name = "_".join(permutation)
            _, _, precision, recall = predict(clf, test, feature_columns=permutation)
            precisions.append(precision)
            recalls.append(recall)
        if np.sum(recalls) != 0:
            full_trained = train_tree(feature_df, permutation)
            model_path = f"data/model/model_{model_name}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(full_trained, f)
            results.append({'permutation': permutation, 'precisions': precisions, 'recalls': recalls,
                            'model_path': model_path})
            prec_mean = np.mean(precisions)
            rec_mean = np.mean(recalls)
            log.info(f"Permutation: {permutation} - P: {prec_mean} - R: {rec_mean}")
        # print(f"Permutation: {permutation} - Score: {score}")
    return results


def kfold_gbt(feature_names, feature_df):
    results = []
    precisions = []
    recalls = []
    kf = KFold(n_splits=5)
    for train_index, test_index in kf.split(feature_df):
        train = feature_df.iloc[train_index]
        test = feature_df.iloc[test_index]
        clf = train_gbt(train, feature_names)
        _, _, precision, recall = predict(clf, test, feature_columns=feature_names)
        precisions.append(precision)
        recalls.append(recall)
    clf = train_gbt(feature_df, feature_names)
    model_path = 'data/model/model_gbt.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)
    results.append({'permutation': 'Gradient Boosting (all)', 'precisions': precisions,
                    'recalls': recalls,
                    'model_path': 'data/model/model_gbt.pkl'})
    prec_mean = np.mean(precisions)
    rec_mean = np.mean(recalls)
    log.info(f"Permutation: Gradient Boosting (all) - P: {prec_mean} - R: {rec_mean}")
    return results


def main():
    warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
    args = parse_args()
    feature_df, feature_names = build_feature_df(args.feature_names)
    log.info(f"Using {len(feature_df)} samples")
    results_trees = kfold_trees(feature_names, feature_df)
    results_gbt = kfold_gbt(feature_names, feature_df)

    results_df = pd.DataFrame(results_trees + results_gbt)

    results_df['recall_mean'] = results_df['recalls'].apply(np.mean)
    results_df['recall_var'] = results_df['recalls'].apply(np.var)
    results_df['precision_mean'] = results_df['precisions'].apply(np.mean)
    results_df['precision_var'] = results_df['precisions'].apply(np.var)

    results_df.sort_values('precision_mean', ascending=False, inplace=True)

    for _, row in results_df.head(10).iterrows():
        msg = f"M: {row['model_path']} stats: P: {row['precision_mean']:.2f} (var:{row['precision_var']:.2f})"
        msg += f"R: {row['recall_mean']:.2f} (var:{row['recall_var']})"
        log.info(msg)
        # write to disk


if __name__ == "__main__":
    main()
