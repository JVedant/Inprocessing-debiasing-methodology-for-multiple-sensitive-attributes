import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
from PIL import Image
from collections import defaultdict
from skimage import exposure

from src import config

class CXR_Multi_Demographic(torch.utils.data.Dataset):
    def __init__(self, traindir, dataframe, transforms):
        """
        Initialize the CXR_Multi_Demographic dataset.
        
        Args:
            traindir (str): Directory containing the images.
            dataframe (pd.DataFrame): DataFrame with image metadata.
            transforms (callable): Image transformations to apply.
        """
        self.traindir = traindir
        self.df = dataframe
        self.transformations = transforms
        self.data = defaultdict(dict)
        demographic = config.DEMOGRAPHICS.split("_")
        
        # Populate the data dictionary
        counter = 0
        for each in range(len(self.df)):
            self.data[counter] = {
                "image_path": os.path.join(self.traindir, self.df.ChexpertSmallPath.iloc[each]),
                "Demographic": {demographic[i]: self.df[demographic[i]].iloc[each] for i in range(len(demographic))},
                "No Finding": self.df["No Finding Clean"].iloc[each],
                "Edema": self.df["Edema"].iloc[each],
                "Pleural Effusion": self.df["Pleural Effusion"].iloc[each],
                "Lung Opacity": self.df["Lung Opacity"].iloc[each],
                "Atelectasis": self.df["Atelectasis"].iloc[each],
            }
            counter += 1

    def __getitem__(self, idx):
        """
        Get a single item from the dataset.
        
        Args:
            idx (int): Index of the item to retrieve.
        
        Returns:
            dict: A dictionary containing the image, target labels, and demographic information.
        """
        item = self.data[idx]
        
        # Load and process the image
        img = Image.open(item['image_path']).convert("L").resize((256, 256), resample=Image.BILINEAR)
        img = np.array(img)
        img = exposure.equalize_hist(img)  # Histogram equalization
        img = (((img - img.min()) / (img.max() - img.min())) * 255).astype(np.uint8)  # Min-Max Scaling
        img = np.stack((img,) * 3)  # Convert to 3-channel image
        img = np.transpose(img, (1, 2, 0))  # Convert to channel-last format
        
        if self.transformations is not None:
            img = self.transformations(img)
        
        # Prepare target labels
        target = torch.tensor([
            item["No Finding"], item["Edema"], item["Pleural Effusion"],
            item["Lung Opacity"], item["Atelectasis"]
        ])
        
        # Prepare demographic labels
        demo_label = {
            demo: config.DEMO_DICT[demo]['Demo_Labels'][item['Demographic'][demo]]
            for demo in item['Demographic']
        }
        
        return {
            "img": img,
            "target": target,
            "demographic": demo_label,
        }

    def __len__(self):
        """
        Get the total number of items in the dataset.
        
        Returns:
            int: The number of items in the dataset.
        """
        return len(self.data)