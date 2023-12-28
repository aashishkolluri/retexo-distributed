"""Implement training process for a worker"""

from typing import Callable, Dict, List, Optional, Tuple
import logging
import dgl # type: ignore
import torch
from torch import nn
from torch.optim import Optimizer
from trainers.base_worker_trainer import BaseWorkerTrainer

logger = logging.getLogger(__name__)


class WorkerTrain(BaseWorkerTrainer):
    """Implement training process for a worker

    Parameters
    ----------
    """

    def __init__(self):
        pass

    def train_model(
        self,
        model: nn.Module,
        traindata: Tuple[dgl.DGLGraph, torch.Tensor, torch.Tensor],
        optimizer: Optimizer,
        loss_function: Callable,
        local_epochs: int = 1,
    ) -> None:
        """Train the model

        Parameters
        ----------
        model : nn.Module
            Model to be trained
        traindata : Tuple[dgl.DGLGraph, torch.Tensor, torch.Tensor]
            DGLGraph, input features and labels for train
        optimizer : Optimizer
            Optimizer for training
        loss_function : Callable
            Loss function
        local_epochs : int, optional
            Number of local epochs, by default 1
        """
        model.train()
        for _ in range(local_epochs):
            self._train_one_epoch(model, traindata, optimizer, loss_function)

    def _train_one_epoch(
        self,
        model: nn.Module,
        traindata: Tuple[dgl.DGLGraph, torch.Tensor, torch.Tensor],
        optimizer: Optimizer,
        loss_function: Callable,
    ) -> None:
        """Train the model for one epoch"""
        sub_g, input_features, output_labels = traindata
        optimizer.zero_grad()
        out = model(sub_g, input_features)
        relevant_nodes = torch.unique(sub_g.edges()[1])
        out = out[relevant_nodes]
        loss = loss_function(out, output_labels)
        loss.backward()

    def evaluate(
        self,
        model: nn.Module,
        data: Tuple[dgl.DGLGraph, torch.Tensor, torch.Tensor],
        metric_fn: List[Callable],
        loss_function: Callable,
    ) -> Dict[str, float]:
        """Evaluate the model"""
        model.eval()
        evaluated_metrics: Dict[str, List[float]] = {f.__name__: [] for f in metric_fn}
        evaluated_metrics["loss"] = []
        sub_g, input_features, output_labels = data
        with torch.no_grad():
            out = model(sub_g, input_features)
            relevant_nodes = torch.unique(sub_g.edges()[1])
            out = out[relevant_nodes]
            loss = loss_function(out, output_labels)
        evaluated_metrics["loss"].append(loss.item())
        for f in metric_fn:
            evaluated_metrics[f.__name__].append(f(out, output_labels))
        return {k: sum(v) / len(v) for k, v in evaluated_metrics.items()}

    def get_embeddings(
        self,
        model: nn.Module,
        data: Tuple[dgl.DGLGraph, torch.Tensor, torch.Tensor],
        emb_shape: Tuple[int, int],
        hidden_layer: Optional[bool]=False
    ) -> torch.Tensor:
        """Get embeddings of all the destination nodes
        in the sorted order of the nids
        """
        model.eval()
        # embeddings_dict = {}
        sub_g, input_features, _ = data
        embeddings = torch.zeros(emb_shape)

        kwargs = {}
        if hidden_layer:
            kwargs["hidden_layer"] = True
        with torch.no_grad():
            out = model(sub_g, input_features, **kwargs)
            relevant_nodes = torch.unique(sub_g.edges()[1])
            out = out[relevant_nodes]
        gids = sub_g.ndata[dgl.NID]

        id_tensor = gids[relevant_nodes].to("cpu")
        embeddings[id_tensor] = out.to("cpu")

        return embeddings
