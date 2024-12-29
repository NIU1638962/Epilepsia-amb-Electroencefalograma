from datasets import SeizuresDataset
from utils import echo
from train_classifier import train_classifier
from dataloaders import create_dataloader
from  backbone import InputLevelFusion, FeatureLevelFusion
from torch.nn import CrossEntropyLoss, optim
import torch.optim as optim
import torch
import os
import gc


DIRECTORY_PATH = ""
DIRECTORY_SAVE_MODELS = "../Trained_models"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    echo(device)

    echo('Reading Dataset...')

    data = SeizuresDataset(DIRECTORY_PATH)

    batch_size = 32

    loader = create_dataloader(data, batch_size)

    echo('Dataset Readed')

    for model_type in ["Input", "Feature"]:

        if(model_type == "Input"):
            model = InputLevelFusion()
        else:
            model = FeatureLevelFusion()

        loss_func = CrossEntropyLoss()

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        num_epochs = 10

        model, log_loss = train_classifier(model, loss_func, device, loader, optimizer, num_epochs)

        torch.save(
                model.state_dict(),
                os.path.join(
                    DIRECTORY_SAVE_MODELS,
                    model_type+"_model.pth",
                ),
            )
        
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()

