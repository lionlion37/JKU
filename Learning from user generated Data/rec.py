import pandas as pd
import numpy as np
import random as rnd

import torch
from torch import nn
from sklearn.metrics.pairwise import cosine_similarity


#########################################
# EXERCISE 3
#########################################


def svd_decompose(inter_matr: np.ndarray, f=50) -> (np.ndarray, np.ndarray):
    """
    inter_matr - np.ndarray - interaction matrix to construct svd from.
    f - int - expected size of embeddings

    returns - U_final, V_final - (as above) user-/item-embeddings of given length f
    """

    U, s, Vh = np.linalg.svd(inter_matr, full_matrices=False)
    U_final = U[:, :f] @ np.diag(s[:f] ** 0.5)  # users x features
    V_final = (np.diag(s[:f] ** 0.5) @ Vh[:f, :]).T  # items x features

    return U_final, V_final


def svd_recommend_to_list(user_ids: list, seen_item_ids: list, U: np.ndarray, V: np.ndarray, topK: int) -> np.ndarray:
    """
    Recommend with svd to selected users

    user_ids - list[int] - ids of target users.
    seen_item_ids - list[list[int]] ids of items already seen by the users (to exclude from recommendation)
    U and V - user- and item-embeddings
    topK - number of recommendations per user to be returned

    returns - np.ndarray - list of lists of ids of recommended items in the order of descending score, for every user
                           make sure the dimensions are correct: [(number of users) x (topK)]
                           use -1 as a place holder item index, when it is impossible to recommend topK items
    """

    items = list(range(V.shape[0]))
    recs = np.full((U.shape[0], topK), -1)

    for user, i_seen in zip(user_ids, seen_item_ids):

        user_embed = U[user]
        scores = np.full((len(items),), -1.0)

        for item in items:
            if item not in i_seen:
                item_embed = V[item]
                score = cosine_similarity(user_embed.reshape(1, -1), item_embed.reshape(1, -1))
                scores[item] = score

        m = min(topK, scores.shape[0])
        recs[user, :m] = (-scores).argsort()[:m]

    return np.array(recs)


class MF(nn.Module):

    def __init__(self, n_users: int, n_items: int, n_factors: int):
        """
        n_users - int - number of users.
        n_items - int - number of items.
        n_factors - int - dimensionality of the latent space.
        """

        super(MF, self).__init__()

        self.embedding_user = nn.Embedding(n_users, n_factors)
        self.embedding_item = nn.Embedding(n_items, n_factors)

    def forward(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        """
        We allow for some flexibility giving lists of ids as inputs:
        if the training data is small we can deal with it in a single forward pass,
        otherwise we could fall back to mini-batches, limiting users and items we pass
        every time.

        user - torch.Tensor - user_ids.
        item - torch.Tensor - item_ids.

        returns - torch.Tensor - Reconstructed Interaction matrix of shape (n_users, n_items).
        """
        u = self.embedding_user(user)
        v = self.embedding_item(item)

        return u @ v.T


def compute_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    logits - torch.Tensor - output of model.
    labels - torch.Tensor - labels / interaction matrix model should learn to reconstruct.

    returns - torch.Tensor - BCELoss over all logits and labels.
    """
    values = nn.Sigmoid()(logits)
    loss = nn.BCELoss()(values, labels)

    return loss


def train(model: nn.Module, train_data_inter: np.ndarray, epochs: int, optimizer, loss_func) -> list:
    """
    model - nn.Module - torch module to train.
    train_data_inter - np.ndarray - interaction matrix of the training data.
    epochs - int - number of epochs to perform.
    optimizer - optim - optimizer for training.
    loss_func - loss function for training.

    returns - list - list of loss values over all epochs.
    """
    losses = []

    model.train()

    user_ids = torch.Tensor(list(range(train_data_inter.shape[0]))).long()
    item_ids = torch.Tensor(list(range(train_data_inter.shape[1]))).long()
    y = torch.Tensor(train_data_inter).long()

    for e in range(epochs):
        optimizer.zero_grad()

        y_hat = model(user_ids, item_ids)

        loss = loss_func(y_hat.unsqueeze(0).float(), y.unsqueeze(0).float())
        loss.backward()
        optimizer.step()

        if e % 100 == 0:
            print("Loss ", e, ": ", loss.item())

        losses.append(loss.item())

    return losses


def itMF_recommend_to_list(user_ids: list, seen_item_ids: list, model, topK=10) -> np.ndarray:
    """
    Recommend with the trained model to selected users

    user_ids - list[int] - ids of target users.
    seen_item_ids - list[list[int]] ids of items already seen by the users (to exclude from recommendation)
    model - trainted factorization model to use for scoring
    topK - number of recommendations per user to be returned

    returns - np.ndarray - list of lists of ids of recommended items in the order of descending score, for every user
                           make sure the dimensions are correct: [(number of user_ids) x (topK)]
                           use -1 as a place holder item index, when it is impossible to recommend topK items
    """

    model.eval()

    items = list(range(model.embedding_item.num_embeddings))
    recs = np.full((len(user_ids), topK), -1)

    with torch.no_grad():

        for i, user, i_seen in zip(range(len(user_ids)), user_ids, seen_item_ids):

            user_embed = model.embedding_user(torch.tensor([user]).long())
            scores = np.full((len(items),), -1.0)

            for item in items:
                if item not in i_seen:
                    item_embed = model.embedding_item(torch.tensor([item]).long())
                    score = cosine_similarity(user_embed, item_embed)
                    scores[item] = score

            m = min(topK, scores.shape[0])
            recs[i, :m] = (-scores).argsort()[:m]

    return np.array(recs)


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


#########################################
# EXERCISE 2
#########################################


def my_jaccard_score(a: np.array, b: np.array) -> float:
    """
    a, b: - vectors of the same length corresponding to the two items

    returns: float - jaccard similarity score for a and b
    """
    a = np.array(a)
    b = np.array(b)

    a_ones = np.where(a == 1)[0]
    b_ones = np.where(b == 1)[0]

    a_and_b_ones = list(set(list(a_ones) + list(b_ones)))
    a_b_ones = np.where(b[a_ones] == 1)[0]

    return len(a_b_ones) / len(a_and_b_ones)


def calculate_sim_scores(inter: np.array, target_vec: np.array) -> np.array:
    """
    inter: np.array - interaction matrix - calculate similarity between each item and the target item (see below)
    target_vec: int - target item vector

    use my_jaccard_similarity function from above

    returns: np.array - similarities between every item from <inter> and <target_vec> in the respective order
    """

    item_similarities = np.zeros((inter.shape[1],))

    # calculate jaccard similarity of every item.
    for item in range(inter.shape[1]):
        inter_items = inter[:, item]
        item_similarities[item] = my_jaccard_score(inter_items, target_vec)

    return np.array(item_similarities)


def get_user_item_score(inter: np.array,
                        target_user: int,
                        target_item: int,
                        n: int = 2) -> float:
    """
    inter: np.array - interaction matrix.
    target_user: target user id
    target_item: int - target item id
    n: int - n closest neighbors to consider for the score prediction

    returns: float - mean of similarity scores
    """

    inter_pred = inter.copy()

    # Get all items which were consumed by the user.
    item_consumed_by_user = inter_pred[target_user, :] == 1
    item_consumed_by_user[target_item] = False

    # get column of the target_item.
    inter_target_item = inter_pred[:, target_item]

    # create a mask to remove the user from the interaction matrix.
    not_user = np.full((inter_pred.shape[0],), True)
    not_user[target_user] = False

    # remove items not interacted with user
    inter_pred = inter_pred[:, item_consumed_by_user]

    # remove user
    inter_pred = inter_pred[not_user]
    inter_target_item = inter_target_item[not_user]

    # get closest items to target_item, which is at the last indices.
    scores = calculate_sim_scores(inter_pred, inter_target_item)

    # get items with the highest scores.
    scores_ids = np.argsort((- scores))
    scores = scores[scores_ids]

    scores = scores[:n]

    if len(scores) > 0:
        # calculate mean of normed scores.
        item_similarities_mean = scores.mean()
    else:
        item_similarities_mean = 0.0

    return item_similarities_mean


def recTopK(inter_matr: np.array,
            user: int,
            top_k: int,
            n: int) -> (np.array, np.array):
    '''
    inter_matr - np.array from the task 1
    user - user_id, integer
    top_k - expected length of the resulting list
    n - number of neighbors to consider

    returns - array of recommendations (sorted in the order of descending scores) & array of corresponding scores
    '''

    scores = np.zeros((inter_matr.shape[1],))

    for item in range(inter_matr.shape[1]):
        if inter_matr[user, item] == 0:
            score = get_user_item_score(inter_matr, user, item, n)
            scores[item] = score

    top_rec = (- scores).argsort()[:top_k]
    scores = scores[top_rec]

    return np.array(top_rec), np.array(scores)


######################################
# EXERCISE 1
######################################


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


def inter_matr_prob(usr_path, itm_path, inter_path, threshold=1):
    '''
    usr_path - string path to the file with users data;
    itm_path - string path to the file with item data;
    inter_path - string path to the file with interaction data;

    returns - 2D np.array, rows - users, columns - items;
    '''

    # reading files as before
    users = pd.read_csv(usr_path, sep='\t', header=None, names=['location', 'age', 'gender', 'date'])
    items = pd.read_csv(itm_path, sep='\t', header=None, names=['artist', 'track'])
    interactions = pd.read_csv(inter_path, sep='\t', header=None, names=['user', 'item', 'num_inters'])

    # estimating the dimensions of the res matrix as before
    n_users = len(users.index)
    n_items = len(items.index)

    # preparing the res matrix
    res_norm = np.zeros([n_users, n_items])

    # filling in the res matrix; not with ones, but with < inter['num_inters'] >...
    # ...as this time we are interested in the number of interactions
    for _, inter in interactions.iterrows():
        curr_user = inter['user']
        curr_item = inter['item']
        res_norm[curr_user, curr_item] = inter['num_inters'] if inter['num_inters'] >= threshold else 0

    # normalization to make the scores for each user to sum up to one
    for i in range(len(res_norm)):
        res_norm[i] = res_norm[i] / res_norm[i].sum()

    return res_norm


def top_k_pop_items(inter_matrix: np.array,
                    top_k: int) -> np.array:
    '''
    :return: top_k of the most popular items.
    '''

    _top_pop = None

    # sum along all users, this way we get a distribution of track popularity over users
    item_pop = inter_matrix.sum(axis=0)

    # make item_pop negative, because argsort sorts in ascending order...
    # ... and we want popular items to be on top;
    # argsort returns indices of the array sorted according to corresponding values;
    # and we take top_K of them
    _top_pop = (-item_pop).argsort()[:top_k]  # Change NONE to something else #

    return _top_pop


def recTopKPop(prepaired_data: np.array,
               user: int,
               top_k: int) -> np.array:
    '''
    prepaired_data - np.array from the task 1;
    user - user_id, integer;
    top_k - expected length of the resulting list;

    returns - list/array of top K popular items that the user has never seen
              (sorted in the order of descending popularity);
    '''

    # global item-popularity distribution:
    item_pop = prepaired_data.sum(axis=0)

    # finding items seen by the user, basicaly indices of non-zero elements ...
    # ... in the interaction array corresponding to the user:
    items_seen = np.nonzero(prepaired_data[user])

    # force values seen by the user to become 'unpopular'
    item_pop[items_seen] = 0

    # same as before, get indices of top_K (new) populat items
    top_pop = (-item_pop).argsort()[:top_k]

    return top_pop
