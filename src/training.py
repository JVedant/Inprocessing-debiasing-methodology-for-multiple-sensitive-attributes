import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
from tqdm import tqdm
# import config
from src.losses import MultiTaskLossWrapper

from src import config

class Engine:
    """
    Engine class for training and validating a debiased multi-task model.
    """

    def __init__(self, model):
        """
        Initialize the Engine.

        Args:
            model: The neural network model to train.
        """
        self.model = model
        self.task_criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(config.BCE_LOSS_WEIGHTS).to(config.DEVICE),
            reduction="none"
        )
        self.adv_criterion = MultiTaskLossWrapper(num_of_task=3)

        params = ([p for p in model.parameters()] + [self.adv_criterion.log_vars])
        self.optimizer = torch.optim.Adam(params=params, lr=config.LR, weight_decay=config.WEIGHT_DECAY)     

    @staticmethod
    def switch_bias_mode(model, mode_val):
        """
        Modify the model's reversal mode.

        Args:
            model: The model to modify.
            mode_val: Boolean indicating whether to enable gradient reversal.
        """
        model.set_reversal(mode_val)

    def process(self, dataloader, is_training=True):
        """
        Process the data for training or validation.

        Args:
            dataloader: DataLoader for the data.
            is_training: Boolean indicating whether to perform training or validation.

        Returns:
            tuple: Average task loss and list of average adversarial losses for each demographic factor.
        """
        self.model.train() if is_training else self.model.eval()
        adv_loss_accumulator = [0.0, 0.0, 0.0]
        task_loss_accumulator = 0.0

        desc = "Training" if is_training else "Validating"
        for batch in tqdm(dataloader, desc=desc):
            images = batch["img"].to(device=config.DEVICE)
            targets = batch["target"].to(device=config.DEVICE)
            dem_labels = batch["demographic"]

            if is_training:
                # First forward pass (without gradient reversal)
                self.switch_bias_mode(self.model, False)
                self.optimizer.zero_grad()
                task_pred, dem_pred = self.model(images)
                task_pred = task_pred.type(torch.float)

                # Calculate losses
                task_loss = self.task_criterion(task_pred, targets)
                adv_loss = self.adv_criterion(dem_pred, dem_labels)
                all_loss = config.LAMBDA * torch.vstack([adv_loss[key] for key in adv_loss.keys()]).T
                tot_loss = torch.cat([task_loss, all_loss], axis=1)
                ov_loss = torch.mean(tot_loss)

                # Backward pass and optimization
                ov_loss.backward()
                self.optimizer.step()

                # Second forward pass (with gradient reversal)
                self.switch_bias_mode(self.model, True)
                _, dem_pred = self.model(images)
                adv_loss = self.adv_criterion(dem_pred, dem_labels)
                all_loss = torch.vstack([adv_loss[key] for key in adv_loss.keys()]).T
                adv_loss_total = torch.mean(all_loss)

                # Backward pass and optimization for adversarial part
                adv_loss_total.backward()
                self.optimizer.step()
            else:
                with torch.no_grad():
                    task_pred, dem_pred = self.model(images)
                    task_pred = task_pred.type(torch.float)
                    task_loss = self.task_criterion(task_pred, targets)
                    adv_loss = self.adv_criterion(dem_pred, dem_labels)

            # Accumulate losses
            for i, each in enumerate(adv_loss.keys()):
                adv_loss_accumulator[i] += adv_loss[each].data.mean().item()
            task_loss_accumulator += task_loss.mean().data.item()

        # Calculate average losses
        adv_loss_accumulator = [each / len(dataloader) for each in adv_loss_accumulator]
        task_loss_accumulator /= len(dataloader)

        return task_loss_accumulator, adv_loss_accumulator

    def train(self, train_dataloader):
        """
        Train the model.

        Args:
            train_dataloader: DataLoader for training data.

        Returns:
            tuple: Average task loss and list of average adversarial losses for each demographic factor.
        """
        return self.process(train_dataloader, is_training=True)

    def validate(self, val_dataloader):
        """
        Validate the model.

        Args:
            val_dataloader: DataLoader for validation data.

        Returns:
            tuple: Average task loss and list of average adversarial losses for each demographic factor.
        """
        return self.process(val_dataloader, is_training=False)