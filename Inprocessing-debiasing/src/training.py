import torch
import torch.nn as nn
import config
from tqdm import tqdm
from losses import MultiTaskLossWrapper


def switch_bias_mode(model,mode_val): 
    """ Modify the config
    """
    model.set_reversal(mode_val)


def trainDeBiased_Multi(model, optimizer, dataloader):
    """
    Train a debiased multi-task model.

    Args:
        model: The neural network model to train.
        optimizer: The optimizer for updating model weights.
        dataloader: DataLoader for training data.

    Returns:
        tuple: Average task loss and list of average adversarial losses for each demographic factor.
    """
    adv_loss_training = [0.0, 0.0, 0.0]  # Initialize adversarial losses for each demographic factor
    task_loss_training = 0.0  # Initialize task loss

    # Define loss functions
    task_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(config.BCE_LOSS_WEIGHTS).to(config.DEVICE), reduction="none")
    adv_criterion = MultiTaskLossWrapper(num_of_task=3)

    for batch in tqdm(dataloader, desc="Training"):
        images = batch["img"].to(device=config.DEVICE)
        targets = batch["target"].to(device=config.DEVICE)
        dem_labels = batch["demographic"]

        # First forward pass (without gradient reversal)
        switch_bias_mode(model, False)
        optimizer.zero_grad()
        model.train()
        task_pred, dem_pred = model(images)
        task_pred = task_pred.type(torch.float)

        # Calculate losses
        task_loss = task_criterion(task_pred, targets)
        adv_loss = adv_criterion(dem_pred, dem_labels)
        all_loss = config.LAMBDA * torch.vstack([adv_loss[key] for key in adv_loss.keys()]).T
        tot_loss = torch.cat([task_loss, all_loss], axis=1)
        ov_loss = torch.mean(tot_loss)

        # Backward pass and optimization
        ov_loss.backward()
        optimizer.step()  # update weights

        # Second forward pass (with gradient reversal)
        switch_bias_mode(model, True)
        _, dem_pred = model(images)
        adv_loss = adv_criterion(dem_pred, dem_labels)
        all_loss = torch.vstack([adv_loss[key] for key in adv_loss.keys()]).T
        adv_loss_total = torch.mean(all_loss)

        # Backward pass and optimization for adversarial part
        adv_loss_total.backward()
        optimizer.step()  # update weights

        # Accumulate losses
        adv_loss_total = 0
        for i, each in zip(range(len(dem_pred)), adv_loss.keys()):
            adv_loss_training[i] += adv_loss[each].data.mean().item()
            adv_loss_total += adv_loss[each]
        task_loss_training += task_loss.mean().data.item()

    # Calculate average losses
    adv_loss_training = [each/len(dataloader) for each in adv_loss_training]
    task_loss_training /= len(dataloader)

    return task_loss_training, adv_loss_training