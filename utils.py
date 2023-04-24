"""
Contains various utility functions for PyTorch model training, saving and loading.
"""
import torch
from pathlib import Path
import os
import csv
from typing import Dict, List, Tuple

def save_model(
    model: torch.nn.Module,
    target_dir: str,
    model_name: str,
    class_names: List[str]
):
    """Saves a PyTorch model to a target directory.

    Args:
        model: A target PyTorch model to save.
        target_dir: A directory for saving the model to.
        model_name: A filename for the saved model. Should include
            either ".pth" or ".pt" as the file extension.
        class_names: A list of all the classes used while training

    Example usage:
        save_model(model=model_0,
                    target_dir="models",
                    model_name="modular_tingvgg_model.pth",
                    class_names=["apple", "banana"])
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                        exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"\n[INFO] Saving model to: {model_save_path}")
    torch.save(
        {
            "model": model,
            "class_names": class_names
        },
        f=model_save_path
    )

def load_model(
    model: torch.nn.Module,
    file_dir: str,
    model_name: str
) -> Dict:
    """
    Saves a PyTorch model to a target directory.

    Args:
        model: A target PyTorch model to load weights of.
        file_dir: A directory for loading the model from.
        model_name: A filename for the saved model. Should include
            either ".pth" or ".pt" as the file extension.

    Example usage:
        save_model(model=model_0,
                    file_dir="models",
                    model_name="modular_tingvgg_model.pth")
    """
    # Create target directory
    file_dir_path = Path(file_dir)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_load_path = file_dir_path / model_name

    # Save the model state_dict()
    print(f"\n[INFO] Loading model from: {model_load_path}")
    model = torch.load(model_load_path)
    return model

def save_data(
    model_name: str,
    data: Dict
) :
    """
    A helper function to save model training data.

    Args:
        model_name (str): takes in the model name
        data (Dict): takes in the data as Dictionary
    """
    result_dir = Path("results") / Path(model_name)
    os.makedirs(result_dir, exist_ok=True)
    csv_columns = ["epoch", "train_loss",  "train_acc", "test_loss", "test_acc"]
    csv_file = result_dir / Path("result.csv")
    total_history = []

    for epoch in range(len(data["epoch"])):
        history = {}
        for column in csv_columns:
            history[column] = data[column][epoch]
        total_history.append(history)
    try:
        if os.path.exists(csv_file):
            os.remove(csv_file)
        with open(csv_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for row in total_history:
                writer.writerow(row)
    except IOError:
        print("I/O error")


def load_data(
    model_name: str,
) -> Dict[str, List]:
    """
    A helper function to save model training data.

    Args:
        model_name (str): takes in the model name
        data (Dict): takes in the data as Dictionary
    """
    result_dir = Path("results") / Path(model_name)
    csv_columns = ["epoch", "train_loss",  "train_acc", "test_loss", "test_acc"]
    result = {
        "epoch": [],
        "train_loss": [],  
        "train_acc": [], 
        "test_loss": [], 
        "test_acc": []
    }
    csv_file = result_dir / Path("result.csv")
    with open(csv_file, 'r') as data:
        for line in csv.DictReader(data):
            for column in csv_columns:
                result[column].append(line[column])
    return result
