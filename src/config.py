import torch

# DEMOGRAPHICS to debias the model for 
DEMOGRAPHICS = "SupportDevices_Race_AgeClass"

# Hyperparameters
LAMBDA = 1e-3  # Lambda for weighting adversarial loss
BCE_LOSS_WEIGHTS = [2.963280766852195, 2.903978678850181, 1.4622514647968494, 1.0015225453835643, 5.586882507869211]
ADV_LOSS_WEIGHTS = {
    "SupportDevices": [1.14335451, 0.88858825],
    "Race": [0.42197229, 2.42269162, 4.5995604],
    "AgeClass": [2.93258973, 1.08588554, 0.57534185]
}
LR = 1e-6  # Learning rate
WEIGHT_DECAY = 1e-2  # Weight decay for optimizer

# Data paths
TRAIN_PATH = "/mnt/storage/Vedant/Data/CheXpert/V1/Data/Chexpert/V4/train.csv"
VALID_PATH = "/mnt/storage/Vedant/Data/CheXpert/V1/Data/Chexpert/V4/valid.csv"
TEST_PATH = f"/home/vedant/Projects/Data/Chexpert/test.csv"
DATASET_DIR = '/mnt/storage'

# Set up logging and model directories
LOG_DIR = f'../Logs/{DEMOGRAPHICS}/lambda{LAMBDA}'
MODEL_DIR = f'../Models/{DEMOGRAPHICS}/lambda{LAMBDA}'

# Image configuration
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
IMAGE_CHANNEL = 3

# Training configuration
TRAIN_BATCH_SIZE = 224
VALID_BATCH_SIZE = 512
NUM_WORKER = 5
NUM_CLASSES = 4

# Set device (GPU if available, otherwise CPU)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Training parameters
EPOCHS = 150
SAVE_WEIGHTS_INTERVAL = 20

# Demographic information dictionary
DEMO_DICT = {
    "Sex": {
        "NUM_CLASSES": 2,
        "Demo_Labels": {'Female': 0, 'Male': 1}
    },
    "SupportDevices": {
        "NUM_CLASSES": 2,
        "Demo_Labels": {0.0: 0, 1.0: 1}
    },
    "Race": {
        "NUM_CLASSES": 3,
        "Demo_Labels": {'White or caucasians': 0, 'Asian': 1, 'Black or African American': 2}
    },
    "AgeClass": {
        "NUM_CLASSES": 3,
        "Demo_Labels": {'Age<=40': 0, '40<Age<=60': 1, '60<Age': 2}
    }
}

# Configuration dictionary to save with model weights
config_json = {
    "DEMOGRAPHICS": DEMOGRAPHICS,
    "TARGET": ["No Finding Clean", "Edema", "Pleural Effusion", "Lung Opacity", "Atelectasis"],
    "LAMBDA": LAMBDA,
    "LOSS_WEIGHTS":{
        "BCE_LOSS_WEIGHTS": BCE_LOSS_WEIGHTS,
        "ADV_LOSS_WEIGHTS": ADV_LOSS_WEIGHTS
    },
    "Optimizer_configs":{
        "LR": LR,
        "WEIGHT_DECAY": WEIGHT_DECAY
    },
    "LOGS": LOG_DIR,
    "MODEL_PATH": MODEL_DIR,
    "INPUT_IMAGE": {
        "IMAGE_HEIGHT": IMAGE_HEIGHT,
        "IMAGE_WIDTH": IMAGE_WIDTH,
        "IMAGE_CHANNEL": IMAGE_CHANNEL,
    },
    "TRAIN_BATCH_SIZE": TRAIN_BATCH_SIZE,
    "VALID_BATCH_SIZE": VALID_BATCH_SIZE,
    "EPOCHS": EPOCHS,
    "DEMO_DICT": DEMO_DICT
}