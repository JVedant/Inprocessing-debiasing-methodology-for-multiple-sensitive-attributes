import torch

from src.run import run
from src import config
from src.training import Engine

from src.models.debiasModels import *


# Main execution block
if __name__ == "__main__":
    # Prepare demographic classes
    num_demo_class = []
    for each in config.DEMOGRAPHICS.split("_"):
        num_demo_class.append(config.DEMO_DICT[each]['NUM_CLASSES'])
    
    # Initialize the model
    model = denseNet121BiasMultiLabel(num_dem_classes=tuple(num_demo_class), num_task_classes=config.NUM_CLASSES)
    model.to(device=config.DEVICE)
    model_name = f"denseNet121BiasMultiLabel"

    # Set up the optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)

    # Create object of engine class
    engine = Engine(model=model, optimizer=optimizer)
    
    # Run the training process
    run(engine, model_name, optimizer)