import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import os
import pandas as pd

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src import config
from src.dataset import CXR_Multi_Demographic

from datetime import datetime

# Main function to run the training process
def run(engine, model_name, epochs=config.EPOCHS):
    # Load training and validation data from CSV files
    train_df = pd.read_csv(config.TRAIN_PATH)
    valid_df = pd.read_csv(config.VALID_PATH)
    
    # Define data transformations for training and validation sets
    train_transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomRotation(degrees=15),
                    transforms.Resize((config.IMAGE_HEIGHT, config.IMAGE_WIDTH)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    valid_transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((config.IMAGE_HEIGHT, config.IMAGE_WIDTH)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    # Create dataset objects
    train_dataset = CXR_Multi_Demographic(traindir=config.DATASET_DIR, dataframe=train_df, transforms=train_transform)
    valid_dataset = CXR_Multi_Demographic(traindir=config.DATASET_DIR, dataframe=valid_df, transforms=valid_transform)

    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKER,
        pin_memory=True
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKER,
        pin_memory=True
    )

    # Get current time for logging and saving
    current_time = datetime.now()

    # Set up directories for saving models and tensorboard logs
    save_dir = os.path.join(config.MODEL_DIR, f'{model_name}/{current_time}')
    writer = SummaryWriter(log_dir=os.path.join(config.LOG_DIR, f'{model_name}/{current_time}'))
    max_loss_ratio = 1e-10

    # Main training loop
    for epoch in range(epochs):
        # Train and validate the model
        task_loss_training, adv_loss_training = engine.train(train_dataloader)
        task_loss_validation, adv_loss_validation = engine.validate(valid_dataloader)

        # Prepare dictionaries for logging
        writer_dict_train = {
            "task_loss": task_loss_training, 
            "adv_loss_SupportDevice": adv_loss_training[0],
            "adv_loss_Race": adv_loss_training[1],
            "adv_loss_AgeClass": adv_loss_training[2],
            "adv_loss": (adv_loss_training[0] + adv_loss_training[1] + adv_loss_training[2]) / 3,
        }
        writer_dict_valid = {
            "task_loss": task_loss_validation, 
            "adv_loss_SupportDevice": adv_loss_validation[0],
            "adv_loss_Race": adv_loss_validation[1],
            "adv_loss_AgeClass": adv_loss_validation[2],
            "adv_loss": (adv_loss_validation[0] + adv_loss_validation[1] + adv_loss_validation[2]) / 3,
        }

        # Calculate saving criteria
        saving_criteria = writer_dict_valid['adv_loss'] / writer_dict_valid['task_loss']

        # Print epoch results
        print(
        f"Epoch: {epoch} | Training Task Loss: {task_loss_training} | Training Adversarial Loss: {adv_loss_training} | Validation Task Loss: {task_loss_validation} | Validation Adversarial Loss: {adv_loss_validation}"
        )

        # Create save directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Save model weights at regular intervals
        if epoch % config.SAVE_WEIGHTS_INTERVAL == 0 and epoch > 0:
            save_config = {
                    "model_name": model_name,
                    "model_weights": engine.model.state_dict(),
                    "epoch": epoch,
                    "optimizer": engine.optimizer,
                    "config_json": config.config_json
                }
            torch.save(save_config, os.path.join(save_dir, f'epoch_{epoch}.pth.tar'))

        # Save best model based on saving criteria
        if saving_criteria > max_loss_ratio:
            print(f"Saving Criteria Increased from {max_loss_ratio} to {saving_criteria}")
            print("Saving the model")
            max_loss_ratio = saving_criteria
            save_config = {
                "model_name": model_name,
                "model_weights": engine.model.state_dict(),
                "epoch": epoch,
                "optimizer": engine.optimizer,
                "config_json": config.config_json
            }
            torch.save(save_config, os.path.join(save_dir, "best_model_config.pth.tar"))

        # Log metrics to tensorboard
        for each in writer_dict_train.keys():
            writer.add_scalar(f"{each}/Train", writer_dict_train[each], epoch)
            writer.add_scalar(f"{each}/Valid", writer_dict_valid[each], epoch)