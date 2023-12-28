"""Implement loss function and evaluation metrics for node classification"""

from typing import List
import dgl # type: ignore
import torch
from torchmetrics import Accuracy, F1Score

from tasks.base_task import BaseMetric, BaseTask


class CustomAccuracy(BaseMetric):
    """Calculate accuracy"""

    def __init__(self):
        self.__name__ = "accuracy"

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate accuracy"""

        return Accuracy(task="multiclass", num_classes=pred.shape[1])(
            pred.detach().cpu(), target.cpu()
        )


class CustomAccuracyMultiLabel(BaseMetric):
    """Calculate accuracy for multilabel classification"""

    def __init__(self):
        self.__name__ = "accuracy"

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate accuracy"""

        return Accuracy(task="multilabel", num_labels=pred.shape[1])(
            pred.detach().cpu(), target.cpu()
        )


class CustomMicroF1Score(BaseMetric):
    """Calculate micro F1 score"""

    def __init__(self):
        self.__name__ = "micro_f1"

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate micro F1 score"""

        return F1Score(task="multiclass", average="micro", num_classes=pred.shape[1])(
            pred.detach().cpu(), target.cpu()
        )


class CustomMicroF1ScoreMultiLabel(BaseMetric):
    """Calculate micro F1 score for multilabel classification"""

    def __init__(self):
        self.__name__ = "micro_f1"

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate micro F1 score"""

        return F1Score(task="multilabel", average="micro", num_labels=pred.shape[1])(
            pred.detach().cpu(), target.cpu()
        )


class NodeClassificationTask(BaseTask):
    """Implement loss function and evaluation metrics for node classification"""

    def __init__(self):
        self.__name__ = "node_classification"

    def get_loss_function(self) -> torch.nn.CrossEntropyLoss:
        """Get the loss function for the task

        Returns
        -------
        torch.nn.CrossEntropyLoss
            Loss function
        """
        return torch.nn.CrossEntropyLoss(reduction="mean")

    def get_evaluation_metrics(self) -> List[BaseMetric]:
        """Get all the evaluation metrics for the task

        Returns
        -------
        List[BaseMetric]
            Evaluation metrics, list of callable classes
        """
        return [CustomAccuracy(), CustomMicroF1Score()]

    def construct_graph_and_features(self, graph: dgl.DGLGraph, node_indices):
        """Return a simple subgraph induced on the inbound edges of nodes"""
        sub_g = graph.in_subgraph(node_indices, relabel_nodes=True)
        dst_nodes = torch.unique(sub_g.edges()[1])
        feat = sub_g.ndata["feat"]
        labels = sub_g.ndata["label"][dst_nodes]
        return sub_g, feat, labels

    def get_tr_val_te_data(self, graph: dgl.DGLGraph, node_dict, feat_tag):
        """Return train test and val data"""
        inner_node_indices = torch.arange(node_dict["inner_node"].int().sum())
        graph.ndata["feat"] = node_dict[feat_tag]
        graph.ndata["label"] = torch.full((len(graph.nodes()),), -100)
        graph.ndata["label"][inner_node_indices] = node_dict["label"]
        train_graph = val_graph = test_graph = graph
        train_indices = inner_node_indices[node_dict["train_mask"]]
        train_subgraph, train_feat, train_labels = self.construct_graph_and_features(
            train_graph, train_indices
        )
        val_indices = inner_node_indices[node_dict["val_mask"]]
        val_subgraph, val_feat, val_labels = self.construct_graph_and_features(
            val_graph, val_indices
        )
        test_indices = inner_node_indices[node_dict["test_mask"]]
        test_subgraph, test_feat, test_labels = self.construct_graph_and_features(
            test_graph, test_indices
        )
        return (
            tuple([train_subgraph, train_feat, train_labels]),
            tuple([val_subgraph, val_feat, val_labels]),
            tuple([test_subgraph, test_feat, test_labels]),
        )


class NodeClassificationMultiLabelTask(BaseTask):
    """Implement loss function and evaluation metrics for node classification"""

    def __init__(self):
        self.__name__ = "node_classification"

    def get_loss_function(self) -> torch.nn.BCEWithLogitsLoss:
        """Get the loss function for the task

        Returns
        -------
        torch.nn.CrossEntropyLoss
            Loss function
        """
        return torch.nn.BCEWithLogitsLoss(reduction="mean")

    def get_evaluation_metrics(self) -> List[BaseMetric]:
        """Get all the evaluation metrics for the task

        Returns
        -------
        List[BaseMetric]
            Evaluation metrics, list of callable classes
        """
        return [CustomAccuracyMultiLabel(), CustomMicroF1ScoreMultiLabel()]

    def construct_graph_and_features(self, graph: dgl.DGLGraph, node_indices):
        """Return a simple subgraph induced on the inbound edges of nodes"""
        sub_g = graph.in_subgraph(node_indices, relabel_nodes=True)
        dst_nodes = torch.unique(sub_g.edges()[1])
        feat = sub_g.ndata["feat"]
        labels = sub_g.ndata["label"][dst_nodes]
        return sub_g, feat, labels

    def get_tr_val_te_data(self, graph: dgl.DGLGraph, node_dict, feat_tag):
        """Return train test and val data"""
        inner_node_indices = torch.arange(node_dict["inner_node"].int().sum())
        graph.ndata["feat"] = node_dict[feat_tag]
        graph.ndata["label"] = torch.full(
            (len(node_dict[dgl.NID]), node_dict["label"].shape[1]),
            -100,
            dtype=torch.float32,
        )
        graph.ndata["label"][inner_node_indices] = node_dict["label"]
        train_graph = val_graph = test_graph = graph
        train_indices = inner_node_indices[node_dict["train_mask"]]
        train_subgraph, train_feat, train_labels = self.construct_graph_and_features(
            train_graph, train_indices
        )
        val_indices = inner_node_indices[node_dict["val_mask"]]
        val_subgraph, val_feat, val_labels = self.construct_graph_and_features(
            val_graph, val_indices
        )
        test_indices = inner_node_indices[node_dict["test_mask"]]
        test_subgraph, test_feat, test_labels = self.construct_graph_and_features(
            test_graph, test_indices
        )
        return (
            tuple([train_subgraph, train_feat, train_labels]),
            tuple([val_subgraph, val_feat, val_labels]),
            tuple([test_subgraph, test_feat, test_labels]),
        )
