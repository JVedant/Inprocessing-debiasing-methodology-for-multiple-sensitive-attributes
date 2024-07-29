import torch
from torch import nn
import torchvision.models as models
import torch.nn.functional as F
from models.reverse_func import grad_reverse


class denseNet121Baseline(torch.nn.Module):
    def __init__(self, num_task_classes):
        """
        Initialize the denseNet121BiasMultiLabelV1 model.

        Args:
            num_task_classes (int): Number of classes for the main task.
            num_dem_classes (int or tuple): Number of classes for each demographic factor.
        """
        super().__init__()
        self.model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        in_features = self.model.classifier.in_features

        # Define the main task classifier
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features, in_features // 4),
            nn.ReLU(),
            nn.Linear(in_features // 4, in_features // 16),
            nn.ReLU(),
            nn.Linear(in_features // 16, num_task_classes),
        )

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: Task classification output.
        """
        out = self.embed(x)
        task_classi = self.model.classifier(out)

        return task_classi

    def embed(self, x):
        """
        Extract features from the input using the DenseNet121 backbone.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Extracted features.
        """
        feats = self.model.features(x)
        out = F.relu(feats, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        return out



class denseNet121BiasMultiLabel(torch.nn.Module):
    def __init__(self, num_task_classes, num_dem_classes):
        """
        Initialize the denseNet121BiasMultiLabelV1 model.

        Args:
            num_task_classes (int): Number of classes for the main task.
            num_dem_classes (int or tuple): Number of classes for each demographic factor.
        """
        super().__init__()
        self.model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        in_features = self.model.classifier.in_features

        # Define the main task classifier
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features, in_features // 4),
            nn.ReLU(),
            nn.Linear(in_features // 4, in_features // 16),
            nn.ReLU(),
            nn.Linear(in_features // 16, num_task_classes),
        )

        # Define the demographic classifiers
        if isinstance(num_dem_classes, tuple):
            # Create separate linear branches based on the length of the tuple
            self.model.dem_classifier = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(in_features, in_features // 4),
                    nn.ReLU(),
                    nn.Linear(in_features // 4, in_features // 16),
                    nn.ReLU(),
                    nn.Linear(in_features // 16, num_classes),
                ) for num_classes in num_dem_classes
            ])
        else:
            # Handle other cases (e.g., single integer)
            self.model.dem_classifier = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(in_features, in_features // 4),
                    nn.ReLU(),
                    nn.Linear(in_features // 4, in_features // 16),
                    nn.ReLU(),
                    nn.Linear(in_features // 16, num_dem_classes),
                )
            ])

        self.debias = False

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: Task classification output and list of demographic classification outputs.
        """
        out = self.embed(x)
        task_classi = self.model.classifier(out)

        if self.debias:
            dem_classi = [dem_branch(grad_reverse(out)) for dem_branch in self.model.dem_classifier]
        else:
            dem_classi = [dem_branch(out.detach()) for dem_branch in self.model.dem_classifier]

        return task_classi, dem_classi

    def embed(self, x):
        """
        Extract features from the input using the DenseNet121 backbone.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Extracted features.
        """
        feats = self.model.features(x)
        out = F.relu(feats, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        return out

    def set_reversal(self, mode):
        """
        Set the debiasing mode.

        Args:
            mode (bool): True to enable gradient reversal, False to disable.
        """
        self.debias = mode