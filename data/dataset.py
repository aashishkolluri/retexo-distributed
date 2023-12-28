"""Implement dataset related functions"""

import logging
import os
import json
import numpy as np
from typing import Dict, List, Optional, Tuple
import dgl # type: ignore
from dgl.data import RedditDataset, YelpDataset, CoraGraphDataset, KarateClubDataset # type: ignore
from dgl.distributed import partition_graph, GraphPartitionBook # type: ignore
from ogb.nodeproppred import DglNodePropPredDataset # type: ignore
import torch
from torch_geometric.datasets import FacebookPagePage, Planetoid, LastFMAsia # type: ignore
from torch_geometric.utils.convert import to_dgl # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore

logger = logging.getLogger(__name__)


def load_ogb_dataset(name, data_path):
    """Load ogb dataset into DGLGraph"""
    dataset = DglNodePropPredDataset(name=name, root=data_path)
    split_idx = dataset.get_idx_split()
    g, label = dataset[0]
    n_node = g.num_nodes()
    node_data = g.ndata
    node_data["label"] = label.view(-1).long()
    node_data["train_mask"] = torch.zeros(n_node, dtype=torch.bool)
    node_data["val_mask"] = torch.zeros(n_node, dtype=torch.bool)
    node_data["test_mask"] = torch.zeros(n_node, dtype=torch.bool)
    node_data["train_mask"][split_idx["train"]] = True
    node_data["val_mask"][split_idx["valid"]] = True
    node_data["test_mask"][split_idx["test"]] = True
    return g


def load_data(dataset_name: str, dataset_dir: str, add_self_loop=False) -> Tuple[dgl.DGLGraph, int, int]:
    """Load dataset

    Parameters
    ----------
    dataset_name : str
        Dataset name
    dataset_dir : str
        Directory to save the dataset
    add_self_loop : bool, optional

    Returns
    -------
    dgl.DGLGraph
        Graph in DGL format
    """
    graph = dgl.DGLGraph()
    if dataset_name == "reddit":
        dataset = RedditDataset(self_loop=True, raw_dir=dataset_dir)
        graph = dataset[0]
    elif dataset_name == "yelp":
        data = YelpDataset(raw_dir=dataset_dir)
        graph = data[0]
        graph.ndata["label"] = graph.ndata["label"].float()
        # TODO: remove the following three lines later (see Issue #4806 of DGL).
        graph.ndata["train_mask"] = graph.ndata["train_mask"].bool()
        graph.ndata["val_mask"] = graph.ndata["val_mask"].bool()
        graph.ndata["test_mask"] = graph.ndata["test_mask"].bool()
        feats = graph.ndata["feat"]
        scaler = StandardScaler()
        scaler.fit(feats[graph.ndata["train_mask"]])
        feats = scaler.transform(feats)
        graph.ndata["feat"] = torch.tensor(feats, dtype=torch.float)
    elif dataset_name == "ogbn-products":
        graph = load_ogb_dataset(name="ogbn-products", data_path=dataset_dir)
    elif dataset_name == "ogbn-arxiv":
        graph = load_ogb_dataset(name="ogbn-arxiv", data_path=dataset_dir)
    elif dataset_name == "ogbn-papers100M":
        graph = load_ogb_dataset(name="ogbn-papers100M", data_path=dataset_dir)
    elif dataset_name == "cora":
        dataset = CoraGraphDataset(raw_dir=dataset_dir)
        graph = dataset[0]
    elif dataset_name == "facebook":
        dataset_dir = os.path.join(dataset_dir, "facebook")
        dataset = FacebookPagePage(root=dataset_dir)
        # get train mask, test mask and validation mask
        if os.path.exists(os.path.join(dataset_dir, "train_mask.npy")):
            train_mask = torch.tensor(np.load(os.path.join(dataset_dir, "train_mask.npy")))
            val_mask = torch.tensor(np.load(os.path.join(dataset_dir, "val_mask.npy")))
            test_mask = torch.tensor(np.load(os.path.join(dataset_dir, "test_mask.npy")))
        else:
            train_mask, val_mask, test_mask = get_masks_fb_page(dataset, 0.7, 0.33)
            # save masks
            np.save(os.path.join(dataset_dir, "train_mask.npy"), train_mask)
            np.save(os.path.join(dataset_dir, "val_mask.npy"), val_mask)
            np.save(os.path.join(dataset_dir, "test_mask.npy"), test_mask)
        graph = to_dgl(dataset[0])
        graph.ndata["feat"] = graph.ndata["x"]
        graph.ndata["label"] = graph.ndata["y"]
        graph.ndata["train_mask"] = train_mask
        graph.ndata["val_mask"] = val_mask
        graph.ndata["test_mask"] = test_mask
    elif dataset_name == "lastfm":
        dataset_dir  = os.path.join(dataset_dir, "lastfm")
        dataset = LastFMAsia(root=dataset_dir)
        # get train mask, test mask and validation mask
        if os.path.exists(os.path.join(dataset_dir, "train_mask.npy")):
            train_mask = torch.tensor(np.load(os.path.join(dataset_dir, "train_mask.npy")))
            val_mask = torch.tensor(np.load(os.path.join(dataset_dir, "val_mask.npy")))
            test_mask = torch.tensor(np.load(os.path.join(dataset_dir, "test_mask.npy")))
        else:
            train_mask, val_mask, test_mask = get_masks_fb_page(dataset, 0.7, 0.33)
            np.save(os.path.join(dataset_dir, "train_mask.npy"), train_mask)
            np.save(os.path.join(dataset_dir, "val_mask.npy"), val_mask)
            np.save(os.path.join(dataset_dir, "test_mask.npy"), test_mask)
        graph = to_dgl(dataset[0])
        graph.ndata["feat"] = graph.ndata["x"]
        graph.ndata["label"] = graph.ndata["y"]
        graph.ndata["train_mask"] = train_mask
        graph.ndata["val_mask"] = val_mask
        graph.ndata["test_mask"] = test_mask
    elif dataset_name in ["cora_torch", "citeseer_torch", "pubmed_torch"]:
        num_split = {
            "cora": [86, 300, 1806],
            "citeseer": [178, 357, 2146],
            "pubmed": [1461, 2190, 13144],
        }
        name_d = dataset_name.split("_")[0]
        dataset = Planetoid(
            root=dataset_dir,
            name=name_d,
            split="random",
            num_train_per_class=num_split[name_d][0],
            num_val=num_split[name_d][1],
            num_test=num_split[name_d][2],
        )
        graph = to_dgl(dataset[0])
        dataset_dir = os.path.join(dataset_dir, name_d)
        if not os.path.exists(os.path.join(dataset_dir, "train_mask.npy")):
            np.save(os.path.join(dataset_dir, "train_mask.npy"), graph.ndata["train_mask"])
            np.save(os.path.join(dataset_dir, "val_mask.npy"), graph.ndata["val_mask"])
            np.save(os.path.join(dataset_dir, "test_mask.npy"), graph.ndata["test_mask"])
        else:
            graph.ndata["train_mask"] = torch.tensor(np.load(os.path.join(dataset_dir, "train_mask.npy")))
            graph.ndata["val_mask"] = torch.tensor(np.load(os.path.join(dataset_dir, "val_mask.npy")))
            graph.ndata["test_mask"] = torch.tensor(np.load(os.path.join(dataset_dir, "test_mask.npy")))
        graph.ndata["feat"] = graph.ndata["x"]
        graph.ndata["label"] = graph.ndata["y"]

    elif dataset_name == "karateclub":
        dataset = KarateClubDataset()
        graph = dataset[0]
        # generate features of all zeros
        # graph.ndata["feat"] = torch.zeros(graph.num_nodes(), 1)
        # make feature same as node id
        graph.ndata["feat"] = torch.arange(graph.num_nodes()).unsqueeze(1).float()
        graph.ndata["train_mask"] = torch.zeros(graph.num_nodes(), dtype=torch.bool)
        graph.ndata["val_mask"] = torch.zeros(graph.num_nodes(), dtype=torch.bool)
        graph.ndata["test_mask"] = torch.zeros(graph.num_nodes(), dtype=torch.bool)
        graph.ndata["train_mask"][torch.tensor([0, 1, 20, 22])] = True
        graph.ndata["val_mask"][torch.tensor([4, 5, 6, 7, 8, 9, 10])] = True
        graph.ndata["test_mask"][
            torch.tensor([2, 3, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 24, 25, 26])
        ] = True
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported")

    # number of features and classes
    n_feat = graph.ndata["feat"].shape[1]  # pylint: disable=no-member
    n_class = -1
    if graph.ndata["label"].dim() == 1:  # pylint: disable=no-member
        n_class = graph.ndata["label"].max().item() + 1  # pylint: disable=no-member
    else:
        n_class = graph.ndata["label"].shape[1]  # pylint: disable=no-member

    graph.edata.clear()
    graph = dgl.remove_self_loop(graph)
    if add_self_loop:
        graph = dgl.add_self_loop(graph)

    return graph, n_feat, n_class


def get_masks_fb_page(dataset, te_tr_split=0.2, val_tr_split=0.2):
    """Get train, test and validation masks for Facebook Page dataset"""
    nnodes = len(dataset[0].x)
    # get train mask, test mask and validation mask
    # get an 80-20 split for train-test
    # get an 80-20 split for train-val in train

    nodes = np.array(range(nnodes))
    train_mask = np.array([False] * nnodes)
    test_mask = np.array([False] * nnodes)
    val_mask = np.array([False] * nnodes)

    rng = np.random.default_rng()
    test_ind = rng.choice(nodes, int(te_tr_split * nnodes), replace=False)
    test_mask[np.array(test_ind)] = True
    rem_ind = []
    for ind in range(nnodes):
        if not ind in test_ind:
            rem_ind.append(ind)
    val_ind = rng.choice(rem_ind, int(val_tr_split * (len(rem_ind))), replace=False)
    val_mask[np.array(val_ind)] = True
    train_mask[~(test_mask | val_mask)] = True

    train_mask = torch.Tensor(train_mask)
    train_mask = train_mask.to(torch.bool)

    val_mask = torch.Tensor(val_mask)
    val_mask = val_mask.to(torch.bool)

    test_mask = torch.Tensor(test_mask)
    test_mask = test_mask.to(torch.bool)

    # Testing
    assert ~(train_mask & test_mask).all()
    assert ~(val_mask & test_mask).all()
    assert ~(train_mask & val_mask).all()

    return train_mask, val_mask, test_mask


def graph_partition(
    graph: dgl.DGLGraph,
    dataset_name: str,
    partition_dir: str,
    num_parts: int,
    part_method: Optional[str] = "random",
    part_obj: Optional[str] = "vol",
) -> None:
    """Partition the graph

    Parameters
    ----------
    graph : dgl.DGLGraph
        Graph in DGL format
    dataset_name : str
        Dataset name
    partition_dir : str
        Directory to save the partitions
    num_parts : int
        Number of partitions
    part_method : Optional[str], optional
        Partition method, by default "random"
    part_obj : Optional[str], optional
        Partition objective, by default "vol"

    Returns
    -------
    None
    """

    # partition the graph
    partition_graph_dir = os.path.join(partition_dir, dataset_name)
    partition_config = os.path.join(partition_graph_dir, f"{dataset_name}.json")

    if not os.path.exists(partition_config):
        with graph.local_scope():
            graph.ndata["in_deg"] = graph.in_degrees()
            graph.ndata["out_deg"] = graph.out_degrees()
            partition_graph(
                graph,
                dataset_name,
                num_parts,
                partition_graph_dir,
                part_method=part_method,
                objtype=part_obj,
            )

    # number of features and classes
    n_feat = graph.ndata["feat"].shape[1]  # pylint: disable=no-member
    n_class = -1
    if graph.ndata["label"].dim() == 1:  # pylint: disable=no-member
        n_class = graph.ndata["label"].max().item() + 1  # pylint: disable=no-member
    else:
        n_class = graph.ndata["label"].shape[1]  # pylint: disable=no-member

    n_train = graph.ndata["train_mask"].int().sum().item()  # pylint: disable=no-member
    n_val = graph.ndata["val_mask"].int().sum().item()  # pylint: disable=no-member
    n_test = graph.ndata["test_mask"].int().sum().item()  # pylint: disable=no-member

    with open(
        os.path.join(partition_graph_dir, "meta.json"), "w", encoding="utf-8"
    ) as f_ptr:
        json.dump(
            {
                "n_feat": n_feat,
                "n_class": n_class,
                "n_train": n_train,
                "n_val": n_val,
                "n_test": n_test,
            },
            f_ptr,
        )


def load_partition(
    partition_dir: str, dataset_name: str, part_id: int
) -> Tuple[dgl.DGLGraph, Dict, int, int, int, int, int, GraphPartitionBook]:
    """Load partitioned graph

    Parameters
    ----------
    partition_dir : str
        Directory with saved partitions
    dataset_name : str
        Dataset name
    part_id : int
        Partition id to load

    Returns
    -------
    Tuple[dgl.DGLGraph, Dict, int, int, int, GraphPartitionBook]
        Graph in DGL format, node features dict, # features, # classes, # train, #val, #test, and partition book
    """

    partition_graph_dir = os.path.join(partition_dir, dataset_name)
    partition_config = os.path.join(partition_graph_dir, f"{dataset_name}.json")

    print(
        f"Loading partition {part_id} of dataset {dataset_name} from {partition_config}"
    )

    (
        sub_graph,
        node_feat,
        _,
        graph_partition_book,
        _,
        node_type,
        _,
    ) = dgl.distributed.load_partition(partition_config, part_id)
    node_type = node_type[0]
    node_feat[dgl.NID] = sub_graph.ndata[dgl.NID]
    if "part_id" in sub_graph.ndata:
        node_feat["part_id"] = sub_graph.ndata["part_id"]

    node_feat["inner_node"] = sub_graph.ndata["inner_node"].bool()
    node_feat["label"] = node_feat[node_type + "/label"]
    node_feat["feat"] = node_feat[node_type + "/feat"]
    node_feat["in_deg"] = node_feat[node_type + "/in_deg"]
    node_feat["out_deg"] = node_feat[node_type + "/out_deg"]
    node_feat["train_mask"] = node_feat[node_type + "/train_mask"].bool()
    node_feat.pop(node_type + "/label")
    node_feat.pop(node_type + "/feat")
    node_feat.pop(node_type + "/in_deg")
    node_feat.pop(node_type + "/out_deg")
    node_feat.pop(node_type + "/train_mask")

    node_feat["val_mask"] = node_feat[node_type + "/val_mask"].bool()
    node_feat["test_mask"] = node_feat[node_type + "/test_mask"].bool()
    node_feat.pop(node_type + "/val_mask")
    node_feat.pop(node_type + "/test_mask")

    sub_graph.ndata.clear()
    sub_graph.edata.clear()

    with open(
        os.path.join(partition_graph_dir, "meta.json"), "r", encoding="utf-8"
    ) as f_ptr:
        partition_meta = json.load(f_ptr)

    n_feat = partition_meta["n_feat"]
    n_class = partition_meta["n_class"]
    n_train = partition_meta["n_train"]
    n_val = partition_meta["n_val"]
    n_test = partition_meta["n_test"]

    return (
        sub_graph,
        node_feat,
        n_feat,
        n_class,
        n_train,
        n_val,
        n_test,
        graph_partition_book,
    )
