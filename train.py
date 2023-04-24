"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os
import torch

import data_setup, engine, model_builder, utils
import argparse

from torchvision import transforms

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--MODEL")
    parser.add_argument("--TRAINED", default=False)
    parser.add_argument("--DEVICE", default="cuda")
    parser.add_argument("--EPOCHS")
    parser.add_argument("--BATCH_SIZE", default=8)
    parser.add_argument("--KERNEL_SIZE", default=3)
    parser.add_argument("--STRIDE", default=8)
    parser.add_argument("--LEARNING_RATE", default=0.001, type=float)
    parser.add_argument("--PADDING", default=1)

    args = parser.parse_args()

    # Setup hyperparameters
    NUM_EPOCHS = int(args.EPOCHS)
    BATCH_SIZE = int(args.BATCH_SIZE)
    KERNEL_SIZE = int(args.KERNEL_SIZE)
    STRIDE = int(args.STRIDE)
    PADDING = int(args.BATCH_SIZE)
    LEARNING_RATE = float(args.LEARNING_RATE)

    # Setup directories
    train_dir = "Datasets/train"
    test_dir = "Datasets/valid"

    # Setup target device
    device = "cuda" if torch.cuda.is_available() else "cpu" 

    # Create transforms
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])


    # Create DataLoaders with help from data_setup.py
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=data_transform,
        batch_size=BATCH_SIZE
    )

    # Create model with help from model_builder.py
    
    model_name = args.MODEL.strip().lower()

    if model_name == "vgg19":
        model = model_builder.VGG19(
            num_classes=len(class_names),
            kernel_size=3,
            padding=1, 
            stride=1
        )
        model_name = "VGG19"
    elif model_name == "resnet18":
        model = model_builder.ResNet18(
            num_classes=len(class_names)
        )
        model_name = "ResNet18"
    
    else :
        raise Exception("Wrong Model.")
    


    trained = args.TRAINED == "True"
    # Load if pre-trained
    if trained:
        model_data = utils.load_model(model = model, file_dir = "models", model_name = f"{model_name}.pth")
        model = model_data["model"]
    
    # Set loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=LEARNING_RATE)

    # Load previous training data if available
    if trained:
        prev_res = utils.load_data(model_name)
    else:
        prev_res = None

    # Start training with help from engine.py
    model.to(device)
    results = engine.train(model=model,
                train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                epochs=NUM_EPOCHS,
                device=device,
                results=prev_res)

    # Save the model with help from utils.py
    utils.save_model(model=model,
                    target_dir="models",
                    model_name=f"{model_name}.pth",
                    class_names=class_names)

    # Save the training data with the help from utils.py
    utils.save_data(
        model_name,
        results
    )
    
