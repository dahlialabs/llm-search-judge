import argparse
import pandas as pd
import numpy as np
import warnings
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import export_text, export_graphviz
from sklearn.model_selection import KFold

import itertools
import graphviz
import pickle


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
    parser.add_argument("--num_test", type=int, default=1000)
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
    print(feature_columns)
    precision = same_label_when_pred.sum() / len(same_label_when_pred)
    recall = len(same_label_when_pred) / len(test)
    print(f"Precision: {precision} - Recall: {recall}")

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


def main():
    warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
    args = parse_args()
    feature_df, feature_names = build_feature_df(args.feature_names)
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
            model_path = f"data/model_{model_name}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(full_trained, f)
            results.append({'permutation': permutation, 'precisions': precisions, 'recalls': recalls,
                            'model_path': model_path})
        # print(f"Permutation: {permutation} - Score: {score}")

    kf = KFold(n_splits=5)
    gbt_precisions = []
    gbt_recalls = []
    for train_index, test_index in kf.split(feature_df):
        train = feature_df.tail(len(feature_df) - args.num_test)
        test = feature_df.head(args.num_test)
        clf = train_gbt(train, feature_names)
        _, _, precision, recall = predict(clf, test, feature_columns=feature_names)
        gbt_precisions.append(precision)
        gbt_recalls.append(recall)
        full_trained = train_gbt(feature_df, feature_names)
        with open("data/model_gbt.pkl", 'wb') as f:
            pickle.dump(full_trained, f)
    results.append({'permutation': 'Gradient Boosting (all)', 'precisions': gbt_precisions,
                    'recalls': gbt_recalls,
                    'model_path': 'data/model_gbt.pkl'})

    results_df = pd.DataFrame(results)

    results_df['recall_mean'] = results_df['recalls'].apply(np.mean)
    results_df['recall_var'] = results_df['recalls'].apply(np.var)
    results_df['precision_mean'] = results_df['precisions'].apply(np.mean)
    results_df['precision_var'] = results_df['precisions'].apply(np.var)

    results_df.sort_values('precision_mean', ascending=False, inplace=True)

    for _, row in results_df.head(10).iterrows():
        print(row['model_path'])
        print(row['permutation'], row['precision_mean'], row['recall_mean'],
              row['precision_var'], row['recall_var'])
        # write to disk


if __name__ == "__main__":
    main()
