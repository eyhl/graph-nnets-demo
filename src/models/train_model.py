import json
import logging
import os
import platform
import sys
from datetime import datetime, timezone
from typing import Any, Optional

import hydra
import torch
import torch.nn as nn
import torch_geometric  # type: ignore
from omegaconf import DictConfig, OmegaConf

from src.data.make_dataset import load_data
from src.models.model import GCN

sys.path.append("..")

log = logging.getLogger(__name__)
print = log.info


def _init_wandb(hparams: Any) -> Optional[Any]:
    if not getattr(hparams, "use_wandb", False):
        return None

    import wandb  # imported lazily

    return wandb.init(
        project=getattr(hparams, "wandb_project", "graph-nnets-demo"),
        entity=getattr(hparams, "wandb_entity", None),
        config=OmegaConf.to_container(hparams, resolve=True),
    )


def evaluate(model: nn.Module, data: torch_geometric.data.Data) -> float:
    """
    Evaluates model on data and returns accuracy.
    :param model: Model to be evaluated
    :param data: Data to evaluate on
    :return: accuracy
    """
    model.eval()
    out = model(data.x, data.edge_index)
    # Use the class with highest probability.
    pred = out.argmax(dim=1)
    # Check against ground-truth labels.
    test_correct = pred[data.test_mask] == data.y[data.test_mask]
    # Derive ratio of correct predictions.
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
    return test_acc


@hydra.main(config_path="../config", config_name="default_config.yaml")
def train(config: DictConfig) -> None:
    """
    Trains the model with hyperparameters in config on train data,
    saves the model and evaluates it on test data.
    :param config: Config file used for Hydra
    :return:
    """
    print(f"configuration: \n {OmegaConf.to_yaml(config)}")
    hparams = config.experiment.hyperparams
    run = _init_wandb(hparams)

    torch.manual_seed(hparams["seed"])
    orig_cwd = hydra.utils.get_original_cwd()

    # Load data
    data = load_data(orig_cwd + "/data/", name="Cora")

    # Model
    model = GCN(
        hidden_channels=hparams["hidden_channels"],
        num_features=hparams["num_features"],
        num_classes=hparams["num_classes"],
        dropout=hparams["dropout"],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams["lr"], weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    epochs = hparams["epochs"]
    train_loss = []

    # Train model
    for epoch in range(epochs):
        # Clear gradients
        optimizer.zero_grad()
        # Perform a single forward pass
        out = model(data.x, data.edge_index)
        # Compute the loss solely based on the training nodes
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        # Derive gradients
        loss.backward()
        # Update parameters based on gradients
        optimizer.step()
        # Append results
        train_loss.append(loss.item())
        # print
        print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")
        if run is not None:
            import wandb

            wandb.log({"train/loss": loss.item(), "epoch": epoch})

    # Save model + metadata
    directory = orig_cwd + "/models/"
    os.makedirs(directory, exist_ok=True)

    checkpoint_path = os.path.join(directory, hparams["checkpoint_name"])
    checkpoint = {
        "num_features": hparams["num_features"],
        "num_classes": hparams["num_classes"],
        "hidden_channels": hparams["hidden_channels"],
        "dropout": hparams["dropout"],
        "state_dict": model.state_dict(),
    }
    torch.save(checkpoint, checkpoint_path)

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": os.path.basename(checkpoint_path),
        "hydra": OmegaConf.to_container(config, resolve=True),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": getattr(torch, "__version__", None),
        "torch_geometric": getattr(torch_geometric, "__version__", None),
        "wandb": bool(run is not None),
    }
    with open(os.path.join(directory, "run.json"), "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, sort_keys=True)
        file.write("\n")

    # Evaluate model
    test_acc = evaluate(model, data)
    print(f"Test accuracy: {test_acc * 100:.2f}%")
    if run is not None:
        import wandb

        wandb.log({"test/accuracy": test_acc})
        run.finish()


if __name__ == "__main__":
    train()
