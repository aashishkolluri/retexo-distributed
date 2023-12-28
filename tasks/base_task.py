"""Implement the base task class."""

from abc import ABC, abstractmethod
from typing import Any, Callable, List, Tuple
import dgl  # type: ignore
from torch import Tensor


class BaseMetric(ABC):
    """Implement the base metric class."""

    @abstractmethod
    def __call__(self, *args, **kwargs) -> float:
        """Calculate the metric

        Returns
        -------
        float
            Metric value
        """


class BaseTask(ABC):
    """Implement the base task class consisting of task specific datasplitting, loss functions and metrics."""

    @abstractmethod
    def get_loss_function(self) -> Callable:
        """Get the loss function for the task

        Returns
        -------
        Callable
            Loss function
        """

    @abstractmethod
    def get_evaluation_metrics(self) -> List[BaseMetric]:
        """Get all the evaluation metrics for the task

        Returns
        -------
        List[BaseMetric]
            Evaluation metrics, list of callable classes
        """

    @abstractmethod
    def get_tr_val_te_data(
        self, graph: dgl.DGLGraph, node_dict: Any, feat_tag: str
    ) -> Tuple[
        Tuple[dgl.DGLGraph, Tensor, Tensor],
        Tuple[dgl.DGLGraph, Tensor, Tensor],
        Tuple[dgl.DGLGraph, Tensor, Tensor],
    ]:
        """Return train test and val data

        Parameters
        ----------
        graph : dgl.DGLGraph
            Graph
        node_dict : Any
            Data dictionary for nodes that contains the train, test and val node indices, features, labels, etc.,
        feat_tag : str
            Feature tag feat_0, feat_1, etc.,

        Returns
        -------
        Tuple[Tuple[dgl.DGLGraph, Tensor, Tensor],Tuple[dgl.DGLGraph, Tensor, Tensor], Tuple[dgl.DGLGraph, Tensor, Tensor]]
            Tuples for train, val, and test data where each tuple contains a DGLGraph, input features and labels
        """
