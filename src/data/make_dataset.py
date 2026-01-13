# -*- coding: utf-8 -*-

import torch_geometric.data  # type: ignore
from torch_geometric.datasets import Planetoid  # type: ignore


def load_data(path: str, name: str) -> torch_geometric.data.Data:
    """Load a PyG dataset and return the single graph.

    For Cora/CiteSeer/PubMed ("Planetoid" datasets), this will download the data
    into `path` if missing, then load it via the stable PyG API.
    """

    dataset = Planetoid(root=path, name=name)
    return dataset[0]
