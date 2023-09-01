Name: Lion Dungl

For this task I implemented ItemKNN using the jaccard distance (1 – jaccard score). Using this algorithm with the cosine distance yielded lower ndcg-scores.

At first, I “trained” the model by calculating the pairwise jaccard distances of each item against each other item using slkearn’s ‘pairwise_distances’ with the metric ‘jaccard’. This resulted in a n_items x n_items distance matrix.

For getting a recommendation for a specific user, I then filtered this distance matrix by selecting all rows with indices of seen items of this user and all columns with indices of unseen items, resulting in a n_seen_items x n_unseen_items matrix (the (n,m)-th entry of this matrix would correspond to the distance between the n-th item, which was seen by the user, and the m-th item, which was not seen).

In the next step, I sorted the columns in descending order. Now, for each unseen item (column) the first entry would be the nearest seen item, etc. Therefore, by selecting the first n-entries per column / unseen item, I’ve got the nearest n seen neighbors of this unseen item. I then was able to calculate the mean distance of each unseen item to the n nearest seen items. Now, for making a recommendation, I just had to select the 15 unseen items with the lowest mean distance.

Using this algorithm, I was able to achieve the same ndcg-scores as with the KNN-algorithm given in rec.py for the sampled_1000_items dataset, but in 1.69s as opposed to 3m25s. For the larger dataset in ex5 I’ve achieved an overall-ndcg-score of around 0.09. I made the observation, that the score was better the higher the hyper-parameter ‘n’ was.
