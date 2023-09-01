import numpy as np
import pandas as pd
import random as rnd


def get_ndcg_score(predictions: np.ndarray, test_interaction_matrix: np.ndarray, topK = 10) -> float:
    """
    predictions - np.ndarray - predictions of the recommendation algorithm for each user.
    test_interaction_matrix - np.ndarray - test interaction matrix for each user.
    topK - int - topK recommendations should be evaluated.

    returns - average ndcg score over all users.
    """
    score = None
    scores = []
    for user_id, pred in enumerate(predictions):
        if sum(test_interaction_matrix[user_id]) == 0 or np.all(pred == -1):  # ignore users w/o interaction in test matrix or w/o predictions
            continue
        # DCG
        current_dcg = 0
        for k, item_id in enumerate(pred[:topK]):
            current_dcg += test_interaction_matrix[user_id, item_id] / np.log2(k + 2)
        # iDCG
        total_interactions = sum(test_interaction_matrix[user_id])
        icg = total_interactions  # ideal cumulative gain
        if total_interactions >= topK:
            icg = topK
        current_idcg = 0
        for k in range(int(icg)):
            current_idcg += 1 / np.log2(k + 2)
        # nDCG
        scores.append(current_dcg / current_idcg)

    score = np.mean(scores)

    return score


def get_recommendations_for_algorithm(config, inter_matrix_train, inter_matrix_test, model) -> np.ndarray:
    """
    inter_matrix_train - np.ndarray - test interaction matrix over all users and items.
    inter_matrix_test -  np.ndarray - train interaction matrix over all users and items.
    config - dict - configuration of this evaluation following the structure:

    config = {
        "algorithm": str - one of ['SVD', 'CF', 'TopPop']
        "inter_train_file_paths": str - inter train files over all splits,
        "inter_test_file_paths": str - inter test files over all splits,
        "user_file_path": str - usr_path,
        "item_file_path": str - itm_path,
        "top_k": int - number of recommendations to evaluate
        "n": int - used for CF.
        "f": int - length of hidden representations for SVD
    }

    returns - np.ndarray - list of recommendations for a specific algorithm in the shape (users, topK)
                           filled with -1 for users NOT in the test set and filled with topK
                           recommendations for users in the test set.
    """
    rec = None

    df_users = pd.read_csv(config['user_file_path'], sep='\t', header=None, names=['location','age','gender', 'date'])
    df_items = pd.read_csv(config['item_file_path'], sep='\t', header=None, names=['artist','track'])
    rec = np.full((len(df_users), config['top_k']), -1)
    user_ids = np.where(np.array([np.all(user == 0) for user in inter_matrix_test]) == False)[0]  # get user_ids of users with at least 1 test-interaction
    
    recommender = model(inter_matrix_train, df_users)
    rec = recommender.predict(user_ids)
    
    """
    if config['algorithm'] == 'SVD':
        U, V = svd_decompose(inter_matrix_train, f=config['f'])

        # get ids of seen items
        seen_item_ids = []
        for interactions in inter_matrix_train[user_ids]:
            seen_item_ids.append(list(np.where(interactions == 1)[0]))

        rec = svd_recommend_to_list(user_ids, seen_item_ids, U, V, config['top_k'])

    elif config['algorithm'] == 'CF':
        for user in user_ids:
            current_recs = recTopK(inter_matrix_train, user, config['top_k'], config['n'])
            rec[user] = current_recs[0]

    elif config['algorithm'] == 'TopPop':
        for user in user_ids:
            current_recs = recTopKPop(inter_matrix_train, user, config['top_k'])
            rec[user] = current_recs
    """

    return rec


def evaluate_predictions(predictions: np.ndarray, test_interaction_matrix: np.ndarray,
                         item_df: pd.DataFrame, topK=10) -> dict:
    """
    This function returns a dictonary with all scores of predictions.

    predictions - np.ndarray - predictions of the algorithm over all users.
    test_interaction_matrix - np.ndarray - test interaction matrix over all users and items.
    item_df - pd.DataFrame - information about each item with columns: 'artist', 'track'
    topK - int - topK prediction should be evaluated

    returns - dict - calculated metric scores, contains keys "ndcg".
    """

    metrics = {}

    ndcg = get_ndcg_score(predictions, test_interaction_matrix, topK)
    metrics['ndcg'] = ndcg

    return metrics


def evaluate_gender(predictions: np.ndarray, test_interaction_matrix: np.ndarray, user_df: pd.DataFrame,
                    item_df: pd.DataFrame, num_users=500, topK=10) -> dict:
    """
    This function will evaluate certain predictions for each gender individually and return a dictionary
    following the structure:

    {'gender_key': {'metric_key': metric_score}}

    predictions - np.ndarray - predictions of the algorithm over all users.
    test_interaction_matrix - np.ndarray - test interaction matrix over all users and items.
    user_df - pd.DataFrame - information about each user with columns: location', 'age', 'gender', 'date'
    item_df - pd.DataFrame - information about each item with columns: 'artist', 'track'
    topK - int - topK prediction should be evaluated

    returns - dict - calculated metric scores for each gender.
    """

    metrics = {}
    for gender in ['m', 'f']:
        gender_ids = np.where(user_df['gender'] == gender)
        metrics[gender] = evaluate_predictions(predictions[gender_ids], test_interaction_matrix[gender_ids], item_df, topK)
    metrics['all'] = evaluate_predictions(predictions, test_interaction_matrix, item_df, topK)

    return metrics


def evaluate_algorithm(config, model) -> dict:
    """
    This function will evaluate a certain algorithm defined with the parameters in config by:
    - going over all test and train files
    - generating the recommendations for each data split
    - calling evaluate gender to get the metrics for each recommendation for each data split

    Then the average score for each gender and metric should be calculated over all data splits and
    a dictionary should be returned following the structure:
    {'gender_key': {'metric_key': avg_metric_score}}

    config - dict - configuration of this evaluation following the structure:

    config = {
        "algorithm": str - one of ['SVD', 'CF', 'TopPop']
        "inter_train_file_paths": str - array of inter train file paths (1 per split),
        "inter_test_file_paths": str - array of inter test file paths (1 per split),
        "user_file_path": str - usr_path,
        "item_file_path": str - itm_path,
        "top_k": int - number of recommendations to evaluate
        "n": int - used for CF.
        "f": int - length of hidden representations for SVD
    }

    returns - dict - average score of each metric for each gender over all data splits.
    """

    metrics = {}
    split_metrics = []
    df_users = pd.read_csv(config['user_file_path'], sep='\t', header=None, names=['location','age','gender', 'date'])
    df_items = pd.read_csv(config['item_file_path'], sep='\t', header=None, names=['artist','track'])

    for train_split_path, test_split_path in zip(config['inter_train_file_paths'], config['inter_test_file_paths']):
        train_split = inter_matr_binary(config['user_file_path'], config['item_file_path'], train_split_path)
        test_split = inter_matr_binary(config['user_file_path'], config['item_file_path'], test_split_path)

        current_recs = get_recommendations_for_algorithm(config, train_split, test_split, model)
        split_metrics.append(evaluate_gender(current_recs, test_split, df_users, df_items, config['top_k']))

    for key in ['m', 'f', 'all']:
        current_dict = {}
        current_dict['ndcg'] = np.mean([curr_split[key]['ndcg'] for curr_split in split_metrics])
        metrics[key] = current_dict

    return metrics


def split_interactions(inter_file='sampled_1000_items_inter.txt',
                       user_file_path='sampled_1000_user.txt',
                       p_i=0.2,
                       p_u=0.1,
                       res_test_file='sampled_1000_items_inter_TEST.txt',
                       res_train_file='sampled_1000_items_inter_TRAIN.txt'):
    '''
    inter_file - string - path to the file with interaction data in LFM2B format;
    proportion - float - proportion of records from inter_file to become the Test Set;
    res_test_file - string - Test records will be saved here;
    res_train_file - string - Train records will be saved here;

    returns - nothing, but saves the two files in LF2B format;
    '''

    df_users = pd.read_csv(user_file_path, sep='\t', header=None, names=['location', 'age', 'gender', 'date'])
    interactions = pd.read_csv(inter_file, sep='\t', header=None, names=['user', 'item', 'num_inters'])

    female_users = df_users[df_users['gender'] == 'f'].index.values.tolist()
    male_users = df_users[df_users['gender'] == 'm'].index.values.tolist()

    user_ids_test = rnd.sample(female_users, int(len(female_users) * p_u)) + rnd.sample(male_users,
                                                                                        int(len(male_users) * p_u))

    size = len(interactions)

    train_indices = []
    test_indices = []

    all_occurrences = []

    for user in user_ids_test:
        occurrences = interactions.loc[interactions['user'] == user].index
        all_occurrences.extend(occurrences)

        l_o = len(occurrences)
        k = int(l_o * p_i)

        if k <= 0:
            k += 1

        random_index = rnd.sample(range(len(occurrences)), k=k)

        index = np.array(occurrences[random_index])
        test_indices.extend(index)

    if len(test_indices) < int(len(all_occurrences) * p_i):
        k = int(len(all_occurrences) * p_i) - len(test_indices)
        occurrences = [o for o in all_occurrences if o not in test_indices]
        random_index = rnd.sample(range(len(occurrences)), k=k)

        test_indices.extend([occurrences[i] for i in random_index])

    train_indices.extend(np.array([idx for idx in range(size) if idx not in test_indices and idx not in train_indices]))

    train_indices.sort()
    test_indices.sort()

    for t_idx in test_indices:
        assert t_idx not in train_indices

    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)

    train = interactions.iloc[train_indices]
    test = interactions.iloc[test_indices]

    print("Train data: ", len(train))
    print("Test data: ", len(test))

    # saving the res files
    # train and test - pd.DataFrames
    train.to_csv(res_train_file, index=False, header=False, sep='\t')
    test.to_csv(res_test_file, index=False, header=False, sep='\t')


def inter_matr_binary(usr_path, itm_path, inter_path, threshold=1):
    '''
    usr_path - string path to the file with users data;
    itm_path - string path to the file with item data;
    inter_path - string path to the file with interaction data;
    threshold - a measure on how many interactions are needed to count as an interaction;

    returns - 2D np.array, rows - users, columns - items;
    '''

    # reading the three data files
    users = pd.read_csv(usr_path, sep='\t', header=None, names=['location', 'age', 'gender', 'date'])
    items = pd.read_csv(itm_path, sep='\t', header=None, names=['artist', 'track'])
    interactions = pd.read_csv(inter_path, sep='\t', header=None, names=['user', 'item', 'num_inters'])

    # getting number of users and items from the respective files to be on the safe side
    n_users = len(users.index)
    n_items = len(items.index)

    # preparing the output matrix
    res = np.zeros([n_users, n_items])

    # for every interaction assign 1 to the respective element of the matrix
    for _, inter in interactions.iterrows():
        curr_user = inter['user']
        curr_item = inter['item']
        res[curr_user, curr_item] = 1 if inter['num_inters'] >= threshold else 0

    return res
