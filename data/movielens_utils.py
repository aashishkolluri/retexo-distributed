
"""Graph builder from pandas dataframes
from https://github.com/dmlc/dgl/blob/master/examples/pytorch/pinsage/builder.py"""
from collections import namedtuple

import dask.dataframe as dd
import dgl
import torch
import scipy.sparse as ssp
import numpy as np

# from pandas.api.types import (
#     is_categorical,
#     is_categorical_dtype,
#     is_numeric_dtype,
# )

__all__ = ["PandasGraphBuilder"]


# This is the train-test split method most of the recommender system papers running on MovieLens
# takes.  It essentially follows the intuition of "training on the past and predict the future".
# One can also change the threshold to make validation and test set take larger proportions.
def train_test_split_by_time(df, timestamp, user):
    df["train_mask"] = np.ones((len(df),), dtype=np.bool_)
    df["val_mask"] = np.zeros((len(df),), dtype=np.bool_)
    df["test_mask"] = np.zeros((len(df),), dtype=np.bool_)
    df = dd.from_pandas(df, npartitions=10)

    def train_test_split(df):
        df = df.sort_values([timestamp])
        if df.shape[0] > 1:
            df.iloc[-1, -3] = False
            df.iloc[-1, -1] = True
        if df.shape[0] > 2:
            df.iloc[-2, -3] = False
            df.iloc[-2, -2] = True
        return df

    meta_df = {
        "user_id": np.int64,
        "movie_id": np.int64,
        "rating": np.int64,
        "timestamp": np.int64,
        "user_id": np.int64,
        "train_mask": bool,
        "val_mask": bool,
        "test_mask": bool,
    }

    df = (
        df.groupby(user, group_keys=False)
        .apply(train_test_split, meta=meta_df)
        .compute(scheduler="processes")
        .sort_index()
    )
    print(df[df[user] == df[user].unique()[0]].sort_values(timestamp))
    return ( # indices of train, val, test
        df["train_mask"].to_numpy().nonzero()[0],
        df["val_mask"].to_numpy().nonzero()[0],
        df["test_mask"].to_numpy().nonzero()[0],
    ) 
    # return (
    #     df["train_mask"],
    #     df["val_mask"],
    #     df["test_mask"],
    # )


def build_train_graph(g, train_indices, utype, itype, etype, etype_rev):
    train_g = g.edge_subgraph(
        {etype: train_indices, etype_rev: train_indices}, relabel_nodes=False
    )

    # copy features
    for ntype in g.ntypes:
        for col, data in g.nodes[ntype].data.items():
            train_g.nodes[ntype].data[col] = data
    for etype in g.etypes:
        for col, data in g.edges[etype].data.items():
            train_g.edges[etype].data[col] = data[
                train_g.edges[etype].data[dgl.EID]
            ]

    return train_g


def build_val_test_matrix(g, val_indices, test_indices, utype, itype, etype):
    n_users = g.num_nodes(utype)
    n_items = g.num_nodes(itype)
    
    # src = user ID (# of local users) dst = movie ID
    val_src, val_dst = g.find_edges(val_indices, etype=etype)
    # val_dst = val_dst - graph_partition_book.local_ntype_offset[1]
    test_src, test_dst = g.find_edges(test_indices, etype=etype)
    # test_dst = test_dst - graph_partition_book.local_ntype_offset[1]
    val_src = val_src.numpy()
    val_dst = val_dst.numpy()
    test_src = test_src.numpy()
    test_dst = test_dst.numpy()
    val_matrix = ssp.coo_matrix(
        (np.ones_like(val_src), (val_src, val_dst)), (n_users, n_items)
    )
    test_matrix = ssp.coo_matrix(
        (np.ones_like(test_src), (test_src, test_dst)), (n_users, n_items)
    )

    return val_matrix, test_matrix


# def _series_to_tensor(series):
#     if is_categorical(series):
#         return torch.LongTensor(series.cat.codes.values.astype("int64"))
#     else:  # numeric
#         return torch.FloatTensor(series.values)


class PandasGraphBuilder(object):
    """Creates a heterogeneous graph from multiple pandas dataframes.

    Examples
    --------
    Let's say we have the following three pandas dataframes:

    User table ``users``:

    ===========  ===========  =======
    ``user_id``  ``country``  ``age``
    ===========  ===========  =======
    XYZZY        U.S.         25
    FOO          China        24
    BAR          China        23
    ===========  ===========  =======

    Game table ``games``:

    ===========  =========  ==============  ==================
    ``game_id``  ``title``  ``is_sandbox``  ``is_multiplayer``
    ===========  =========  ==============  ==================
    1            Minecraft  True            True
    2            Tetris 99  False           True
    ===========  =========  ==============  ==================

    Play relationship table ``plays``:

    ===========  ===========  =========
    ``user_id``  ``game_id``  ``hours``
    ===========  ===========  =========
    XYZZY        1            24
    FOO          1            20
    FOO          2            16
    BAR          2            28
    ===========  ===========  =========

    One could then create a bidirectional bipartite graph as follows:
    >>> builder = PandasGraphBuilder()
    >>> builder.add_entities(users, 'user_id', 'user')
    >>> builder.add_entities(games, 'game_id', 'game')
    >>> builder.add_binary_relations(plays, 'user_id', 'game_id', 'plays')
    >>> builder.add_binary_relations(plays, 'game_id', 'user_id', 'played-by')
    >>> g = builder.build()
    >>> g.num_nodes('user')
    3
    >>> g.num_edges('plays')
    4
    """

    def __init__(self):
        self.entity_tables = {}
        self.relation_tables = {}

        self.entity_pk_to_name = (
            {}
        )  # mapping from primary key name to entity name
        self.entity_pk = {}  # mapping from entity name to primary key
        self.entity_key_map = (
            {}
        )  # mapping from entity names to primary key values
        self.num_nodes_per_type = {}
        self.edges_per_relation = {}
        self.relation_name_to_etype = {}
        self.relation_src_key = {}  # mapping from relation name to source key
        self.relation_dst_key = (
            {}
        )  # mapping from relation name to destination key

    def add_entities(self, entity_table, primary_key, name):
        entities = entity_table[primary_key].astype("category")
        if not (entities.value_counts() == 1).all():
            raise ValueError(
                "Different entity with the same primary key detected."
            )
        # preserve the category order in the original entity table
        entities = entities.cat.reorder_categories(
            entity_table[primary_key].values
        )

        self.entity_pk_to_name[primary_key] = name
        self.entity_pk[name] = primary_key
        self.num_nodes_per_type[name] = entity_table.shape[0]
        self.entity_key_map[name] = entities
        self.entity_tables[name] = entity_table

    def add_binary_relations(
        self, relation_table, source_key, destination_key, name
    ):
        src = relation_table[source_key].astype("category")
        src = src.cat.set_categories(
            self.entity_key_map[
                self.entity_pk_to_name[source_key]
            ].cat.categories
        )
        dst = relation_table[destination_key].astype("category")
        dst = dst.cat.set_categories(
            self.entity_key_map[
                self.entity_pk_to_name[destination_key]
            ].cat.categories
        )
        if src.isnull().any():
            raise ValueError(
                "Some source entities in relation %s do not exist in entity %s."
                % (name, source_key)
            )
        if dst.isnull().any():
            raise ValueError(
                "Some destination entities in relation %s do not exist in entity %s."
                % (name, destination_key)
            )

        srctype = self.entity_pk_to_name[source_key]
        dsttype = self.entity_pk_to_name[destination_key]
        etype = (srctype, name, dsttype)
        self.relation_name_to_etype[name] = etype
        self.edges_per_relation[etype] = (
            src.cat.codes.values.astype("int64"),
            dst.cat.codes.values.astype("int64"),
        )
        self.relation_tables[name] = relation_table
        self.relation_src_key[name] = source_key
        self.relation_dst_key[name] = destination_key

    def build(self):
        # Create heterograph
        graph = dgl.heterograph(
            self.edges_per_relation, self.num_nodes_per_type
        )
        return graph
