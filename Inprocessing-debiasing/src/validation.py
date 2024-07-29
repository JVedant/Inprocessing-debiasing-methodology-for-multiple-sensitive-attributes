import torch
import torch.nn as nn
import config
from tqdm import tqdm
from losses import MultiTaskLossWrapper

def validDeBiased_Multi(model, dataloader):
    """
    Validate a debiased multi-task model.

    Args:
        model: The neural network model to validate.
        dataloader: DataLoader for validation data.

    Returns:
        tuple: Average task loss and list of average adversarial losses for each demographic factor.
    """
    adv_loss_validation = [0.0, 0.0, 0.0]  # Initialize adversarial losses for each demographic factor
    task_loss_validation = 0.0  # Initialize task loss

    # Define adversarial loss criterion
    adv_criterion = MultiTaskLossWrapper(num_of_task=3)

    with torch.no_grad():  # Disable gradient computation
        for batch in tqdm(dataloader, desc="Validation"):
            images = batch["img"].to(device=config.DEVICE)
            targets = batch["target"].to(device=config.DEVICE)
            dem_labels = batch["demographic"]

            model.eval()  # Set model to evaluation mode

            # Forward pass
            task_pred, dem_pred = model(images)

            # Calculate task loss
            task_loss = nn.BCEWithLogitsLoss()(task_pred.type(torch.float), targets)

            # Calculate adversarial losses
            adv_loss = adv_criterion(dem_pred, dem_labels)

            # Accumulate losses
            for i, each in zip(range(len(dem_pred)), adv_loss.keys()):
                adv_loss_validation[i] += adv_loss[each].mean().data.item()
            task_loss_validation += task_loss.mean().data.item()

    # Calculate average losses
    adv_loss_validation = [each/len(dataloader) for each in adv_loss_validation]
    task_loss_validation /= len(dataloader)

    return task_loss_validation, adv_loss_validation