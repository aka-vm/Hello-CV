# %% Imports
import torch
import torch.nn as nn
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torchinfo import summary


import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import pathlib

# %% Constants
PROJECT_PATH =  pathlib.Path(".") / "MNIST-Digit 8k params"
BATCH_SIZE = 64
NUM_WORKERS = 2     # should not be more than os.cpu_count() // 2
TOTAL_EPOCHS = 25

# %% Utils
def get_train_val_test_dataloaders() -> list[DataLoader]:
    train_data = datasets.MNIST(
        root=PROJECT_PATH,
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )
    train_data, val_data = random_split(train_data, (52000, 8000))
    test_data = datasets.MNIST(
        root=PROJECT_PATH,
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )

    train_dataloader    = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_dataloader      = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_dataloader     = DataLoader(test_data,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    return train_dataloader, val_dataloader, test_dataloader


def evaluate(
    network: nn.Module,
    loss_fn: nn.Module,
    data_loader: DataLoader,
) -> dict[str, float]:
    all_pred = np.empty(0)
    all_target = np.empty(0)

    network.eval()
    total_loss = 0.0
    with torch.no_grad():
        for i, (images, target) in enumerate(data_loader):
            output = network(images)

            loss = loss_fn(output, target)
            total_loss += loss.item()

            pred = torch.argmax(output, 1)
            all_pred = np.concatenate((all_pred, pred))
            all_target = np.concatenate((all_target, target))

    total_loss /= len(data_loader)
    accuracy = accuracy_score(all_target, all_pred)

    return {
        "loss": total_loss,
        "accuracy": accuracy,
    }

# %% Network
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, 1),      # 1x28x28 -> 8x26x26
            nn.ReLU(),
            nn.Conv2d(8, 8, 3, 1),      # 8x26x26 -> 8x24x24
            nn.ReLU(),
            nn.MaxPool2d(2, 2),         # 8x24x24 -> 8x12x12
        )
        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, 1),     #  8x12x12 -> 16x10x10
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1),    # 16x10x10 -> 32x8x8
            nn.ReLU(),
            nn.MaxPool2d(4, 4),         # 32x8x8 -> 32x2x2
        )
        self.out = nn.Linear(4*32, 10)  # 128->10

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)

        x = x.view(x.size(0), -1)       # Flatten Layer
        output = self.out(x)
        return output

    def summary(self, verbose: int=0, **kwargs):
        """
        returns the summary of the model.
        """
        return summary(
            self,
            input_size=(1, 28, 28),
            batch_dim=0,
            col_names = ("input_size", "output_size", "num_params", "kernel_size"),
            verbose = verbose,
            **kwargs
            )

# %% training loop
def train(
    num_epochs: int,
    network: nn.Module,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader=None,
) -> dict[int, dict]:

    print("Training The Network...")
    training_logs = {}
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}: ")
        loop = tqdm(train_dataloader)
        network.train()
        # Train Loop
        total_loss = 0.0
        for i, (images, target) in enumerate(loop):
            optimizer.zero_grad()

            pred = network(images)
            loss = loss_fn(pred, target)

            loss.backward()
            optimizer.step()

            loop_loss = loss.item()
            total_loss += loop_loss
            loop.set_postfix(loss=loop_loss)

        # Evaluate -
        train_metrices = evaluate(network, loss_fn, train_dataloader)
        val_metrices = evaluate(network, loss_fn, val_dataloader)

        print(f"    Train Loss - {train_metrices['loss']:.4f}; Validation Loss - {val_metrices['loss']:.4f}")
        print(f"    Train Accuracy - {train_metrices['accuracy']*100:.2f}%; Validation Accuracy - {val_metrices['accuracy']*100:.2f}%")

        # Callbacks - Logger, checkpoint, lr-decay
        logs = {f"train_{metric}": value for metric, value in train_metrices.items()}
        logs |= {f"val_{metric}": value for metric, value in val_metrices.items()}
        training_logs[epoch] = logs

    print("Training Completed")
    return training_logs

# %% Main Function
def main() -> None:
    train_dataloader, val_dataloader, test_dataloader = get_train_val_test_dataloaders()

    cnn_network = CNN()
    print(cnn_network.summary())
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn_network.parameters(), lr=0.0005)

    logs = train(
        num_epochs=TOTAL_EPOCHS,
        network=cnn_network,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
    )
    best_loss_epoch = min(logs, key=lambda x: logs[x]["val_loss"])
    best_loss = logs[best_loss_epoch]["val_loss"]
    best_acc_epoch = max(logs, key=lambda x: logs[x]["val_accuracy"])
    best_acc = logs[best_acc_epoch]["val_accuracy"]

    testing_logs = evaluate(cnn_network, loss_fn, test_dataloader)

    print("="*50)
    print("Training Highlights")
    print(f"  Epoch {best_acc_epoch}/{TOTAL_EPOCHS} Validation Accuracy of {best_acc*100:.2f}%")
    print(f"  Epoch {best_loss_epoch}/{TOTAL_EPOCHS} Validation Loss of {best_loss:.4f}")
    print("="*50)
    print("Final Test")
    print(f"  Test Loss - {testing_logs['loss']:.4f}")
    print(f"  Test Accuracy - {100*testing_logs['accuracy']:.2f}%")
    print("="*50)


#%% Run Main
if __name__ == "__main__":
    main()


# %%
