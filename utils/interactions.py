"""
Classes describing datasets of user-item interactions. Instances of these
are returned by dataset-fetching and dataset-processing functions.
"""

import numpy as np
import scipy.sparse as sp


class Interactions(object):
    """
    Interactions object. Contains (at a minimum) pair of user-item
    interactions, but can also be enriched with ratings, timestamps,
    and interaction weights.

    For *implicit feedback* scenarios, user ids and item ids should
    only be provided for user-item pairs where an interaction was
    observed. All pairs that are not provided are treated as missing
    observations, and often interpreted as (implicit) negative
    signals.

    For *explicit feedback* scenarios, user ids, item ids, and
    ratings should be provided for all user-item-rating triplets
    that were observed in the dataset.

    Parameters
    ----------

    user_ids: array of np.int32
        array of user ids of the user-item pairs
    item_ids: array of np.int32
        array of item ids of the user-item pairs
    ratings: array of np.float32, optional
        array of ratings
    timestamps: array of np.int32, optional
        array of timestamps
    weights: array of np.float32, optional
        array of weights
    num_users: int, optional
        Number of distinct users in the dataset.
        Must be larger than the maximum user id
        in user_ids.
    num_items: int, optional
        Number of distinct items in the dataset.
        Must be larger than the maximum item id
        in item_ids.

    Attributes
    ----------

    user_ids: array of np.int32
        array of user ids of the user-item pairs
    item_ids: array of np.int32
        array of item ids of the user-item pairs
    ratings: array of np.float32, optional
        array of ratings
    timestamps: array of np.int32, optional
        array of timestamps
    weights: array of np.float32, optional
        array of weights
    num_users: int, optional
        Number of distinct users in the dataset.
    num_items: int, optional
        Number of distinct items in the dataset.
    """

    def __init__(self, user_ids, item_ids,
                 ratings=None,
                 timestamps=None,
                 weights=None,
                 num_users=None,
                 num_items=None):

        self.num_users = num_users or int(user_ids.max() + 1)
        self.num_items = num_items or int(item_ids.max() + 1)

        self.user_ids = user_ids
        self.item_ids = item_ids
        self.ratings = ratings
        self.timestamps = timestamps
        self.weights = weights

        self._check()

    def __repr__(self):
        return f'<Interactions dataset ({self.num_users} users x {self.num_items} items x {len(self)} interactions)>'

    def __len__(self):
        return len(self.user_ids)

    def _check(self):
        if self.user_ids.max() >= self.num_users:
            raise ValueError('Maximum user id greater than declared number of users.')
        if self.item_ids.max() >= self.num_items:
            raise ValueError('Maximum item id greater than declared number of items.')

        num_interactions = len(self.user_ids)

        for name, value in (('item IDs', self.item_ids), ('ratings', self.ratings),
                            ('timestamps', self.timestamps), ('weights', self.weights)):
            if value is None:
                continue

            if len(value) != num_interactions:
                raise ValueError(f'Invalid {name} dimensions: length must be equal to number of interactions')

    def tocoo(self):
        """
        Transform to a scipy.sparse COO matrix.
        """

        row = self.user_ids
        col = self.item_ids
        data = self.ratings if self.ratings is not None else np.ones(len(self))

        return sp.coo_matrix((data, (row, col)), shape=(self.num_users, self.num_items))

    def tocsr(self):
        """
        Transform to a scipy.sparse CSR matrix.
        """

        return self.tocoo().tocsr()