import torch
import torch.nn as nn
import numpy as np
import config
import pdb

# Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics
class MultiTaskLossWrapper(nn.Module):
    def __init__(self, num_of_task):
        """
        Initialize the MultiTaskLossWrapper.

        Args:
            num_of_task (int): Number of tasks (demographic factors).
        """
        super(MultiTaskLossWrapper, self).__init__()
        self.log_vars = nn.Parameter(torch.zeros((num_of_task, )))
        
        # Initialize loss functions for each demographic factor with weighted cross-entropy
        self.criterion_supportDevice = nn.CrossEntropyLoss(weight=torch.tensor(config.ADV_LOSS_WEIGHTS['SupportDevices']).to(config.DEVICE), reduction="none")
        self.criterion_race = nn.CrossEntropyLoss(weight=torch.tensor(config.ADV_LOSS_WEIGHTS['Race']).to(config.DEVICE), reduction="none")
        self.criterion_ageclass = nn.CrossEntropyLoss(weight=torch.tensor(config.ADV_LOSS_WEIGHTS['AgeClass']).to(config.DEVICE), reduction="none")

    def forward(self, pred_dems, ground_dems):
        """
        Compute the multi-task loss.

        Args:
            pred_dems (list): Predicted demographics.
            ground_dems (dict): Ground truth demographics.

        Returns:
            dict: Computed losses for each demographic factor.
        """
        if torch.is_grad_enabled():
            # Compute losses using weighted cross-entropy during training
            loss_supportdevice = self.criterion_supportDevice(pred_dems[0].type(torch.float), ground_dems['SupportDevices'].to(config.DEVICE))
            loss_race = self.criterion_race(pred_dems[1].type(torch.float), ground_dems['Race'].to(config.DEVICE))
            loss_ageclass = self.criterion_ageclass(pred_dems[2].type(torch.float), ground_dems['AgeClass'].to(config.DEVICE))
        else:
            # Compute losses using standard cross-entropy during evaluation
            loss_supportdevice = nn.CrossEntropyLoss(reduction="none")(pred_dems[0].type(torch.float), ground_dems['SupportDevices'].to(config.DEVICE))
            loss_race = nn.CrossEntropyLoss(reduction="none")(pred_dems[1].type(torch.float), ground_dems['Race'].to(config.DEVICE))
            loss_ageclass = nn.CrossEntropyLoss(reduction="none")(pred_dems[2].type(torch.float), ground_dems['AgeClass'].to(config.DEVICE))

        # Apply learned weights to the losses
        precision_supportdevice = torch.exp(-self.log_vars[0])
        loss_supportdevice = precision_supportdevice * loss_supportdevice + self.log_vars[0]

        precision_race = torch.exp(-self.log_vars[1])
        loss_race = precision_race * loss_race + self.log_vars[1]

        precision_ageclass = torch.exp(-self.log_vars[2])
        loss_ageclass = precision_ageclass * loss_ageclass + self.log_vars[2]

        # Combine losses into a dictionary
        losses = {
            'SupportDevice': loss_supportdevice,
            'Race': loss_race,
            'AgeClass': loss_ageclass,
        }

        return losses