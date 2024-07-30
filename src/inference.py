import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from torchvision import transforms
import pandas as pd
from sklearn import metrics

from dataset import CXR_Multi_Demographic
from models import debiasModels

import config


def inference(model, dataloader):
    """
    Perform inference on a dataset using a trained model.

    Args:
        model (torch.nn.Module): The trained model to use for inference.
        dataloader (torch.utils.data.DataLoader): DataLoader containing the dataset to perform inference on.

    Returns:
        list: A list of prediction tensors for each batch.
    """
    preds = []
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        for batch in tqdm(dataloader, desc="Inference"):
            image = batch["img"].to(config.DEVICE)  # Move image to the specified device
            y_preds, _ = model(image)  # Forward pass through the model
            y_preds = torch.sigmoid(y_preds)  # Apply sigmoid to get probabilities
            preds.append(y_preds.cpu())  # Store predictions on CPU
    return preds

def get_best_t(y_true, scores):
    """
    Find the optimal threshold for binary classification using the ROC curve.

    This function calculates the ROC curve and finds the threshold that minimizes
    the Euclidean distance to the perfect classification point (0, 1).

    Args:
        y_true (array-like): True binary labels.
        scores (array-like): Target scores, can either be probability estimates or non-thresholded decision values.

    Returns:
        tuple: A tuple containing:
            - float: The optimal threshold.
            - tuple: The (FPR, TPR) coordinates of the optimal point on the ROC curve.
    """
    # Calculate ROC curve
    fpr, tpr, thresholds = metrics.roc_curve(y_true, scores)
    
    # Combine FPR and TPR into a single array
    arr = np.array((fpr, tpr)).T
    
    # Find the index of the point closest to (0,1)
    ideal_idx = np.sum((arr - np.array((0, 1)))**2, axis=1).argmin()
    
    # Get the optimal threshold and its corresponding point on the ROC curve
    t = thresholds[ideal_idx]
    ideal_point = (fpr[ideal_idx], tpr[ideal_idx])
    
    return t, ideal_point


if __name__ == "__main__":
    # Load the test dataset
    test_df = pd.read_csv('/home/vedant/Projects/Data/Chexpert/V4/test.csv')

    # Define image transformations
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((config.IMAGE_HEIGHT, config.IMAGE_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create dataset
    dataset = CXR_Multi_Demographic(traindir='/mnt/storage', dataframe=test_df, transforms=transform)

    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
    )

    # Initialize the model
    model = debiasModels.denseNet121BiasMultiLabelV1(num_dem_classes=(2, 3, 3), num_task_classes=5)

    # Load model weights
    checkpoint = torch.load('MODEL-PATH')
    model.load_state_dict(checkpoint['model_weights'])

    # Move model to the specified device (GPU if available)
    model = model.to(config.DEVICE)

    # Perform inference
    predictions = inference(model, dataloader)

    # Convert predictions to a list of lists
    f_list = []
    for i in range(predictions.size(0)):
        for j in range(predictions.size(1)):
            try:
                pred = predictions[i][j].tolist()
                f_list.append(pred)
            except:
                break  # Break if there's an error (e.g., end of predictions)

    # Define column names for prediction probabilities
    Column = ['No Finding_pred_prob', 'Edema_pred_prob', 'Pleural Effusion_pred_prob', 'Lung Opacity_pred_prob', 'Atelectasis_pred_prob']

    # Create a new dataframe with prediction probabilities
    new_df = pd.DataFrame(f_list)
    new_df.columns = Column

    # Concatenate the original dataframe with prediction probabilities
    pred_df = pd.concat([test_df, new_df], axis=1)

    # Initialize prediction columns with -1
    pred_df['No Finding_preds'] = -1
    pred_df['Edema_preds'] = -1
    pred_df['Pleural Effusion_preds'] = -1
    pred_df['Lung Opacity_preds'] = -1
    pred_df['Atelectasis_preds'] = -1

    # Define target columns
    target_columns = ['No Finding', 'Edema', 'Pleural Effusion', 'Lung Opacity', 'Atelectasis']

    # Calculate optimal thresholds for each target
    thresholds = [
        get_best_t(pred_df['No Finding'].values, pred_df['No Finding_pred_prob'].values)[0],
        get_best_t(pred_df['Edema'].values, pred_df['Edema_pred_prob'].values)[0],
        get_best_t(pred_df['Pleural Effusion'].values, pred_df['Pleural Effusion_pred_prob'].values)[0],
        get_best_t(pred_df['Lung Opacity'].values, pred_df['Lung Opacity_pred_prob'].values)[0],
        get_best_t(pred_df['Atelectasis'].values, pred_df['Atelectasis_pred_prob'].values)[0]
    ]

    # Apply thresholds to get final predictions
    for col, thr in zip(target_columns, thresholds):
        pred_df[f'{col}_preds'] = (pred_df[f'{col}_pred_prob'] >= thr).astype(int)

    # Save results to CSV
    pred_df.to_csv('outputs.csv', index=False)