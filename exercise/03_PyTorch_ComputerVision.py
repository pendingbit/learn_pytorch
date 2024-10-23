# Import PyTorch
import torch
from torch import nn

# Import torchvision
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor

# Import matplotlib for visualization
import matplotlib.pyplot as plt

# Import dataloader
from torch.utils.data import DataLoader

# Check versions
print(f"PyTorch version: {torch.__version__}\ntorchvision version: {torchvision.__version__}")


######################################## download data ########################################
# Setup training data
train_data = datasets.FashionMNIST(
        root="data", # where to download data to ?
        train=True, # get training data
        download=True, # download data if it doesn't exist on disk
        transform=ToTensor(), # images come as PIL format, we want to turn into Torch tensors
        target_transform=None # you can transform labels as well
)

# Setup testing data
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

# See first training sample
image, label = train_data[0]
class_names = train_data.classes
print(f"train_data type is {type(train_data)}")
print(f"the len of train data is {len(train_data.data)} \n the len of test data is {len(test_data.data)}")
print(f"the train_data[0] image shape is {image.shape} and the label is {label}")
print(f"class name is {class_names}")

# Visualizing our data
plt.imshow(image.squeeze()) # image shape is [1, 28, 28](colour channels, height, width)
plt.title(label)
plt.imshow(image.squeeze(), cmap="gray")
plt.title(class_names[label])

# Visualizing more images
fig = plt.figure(figsize=(9, 9))
rows , cols = 4, 4
for i in range(1, rows * cols + 1):
    random_idx = torch.randint(0, len(train_data), size=[1]).item()
    img, label = train_data[random_idx]
    fig.add_subplot(rows, cols, i)
    plt.imshow(img.squeeze(), cmap="gray")
    plt.title(class_names[label])
    plt.axis(False)

######################################### dataloader with 32 batch size ############################################

# Setup batch size hyperparameter
BATCH_SIZE = 32

# Turn datasets into inerables (batches)
train_dataloader = DataLoader(train_data, # dataset to trun into iterable
                              batch_size=BATCH_SIZE, # how many samples per batch
                              shuffle=True # shuffle data every epoch
                              )

test_dataloader = DataLoader(test_data,
                             batch_size=BATCH_SIZE,
                             shuffle=False
                             )

# Let's check out what we've created
print(f"Dataloaders: {train_dataloader, test_dataloader}")
print(f"Length of train dataloader: {len(train_dataloader)} batches of {BATCH_SIZE}")
print(f"Length of test dataloader: {len(test_dataloader)} batches of {BATCH_SIZE}")
                    
# Checkout what's inside the training dataloader
train_features_batch, train_labels_batch = next(iter(train_dataloader))
print(f"train_features_batch shape is {train_features_batch.shape}")
print(f"train_labels_batch shape is {train_labels_batch.shape}")


# Create a flatten layer
flatten_model = nn.Flatten() # all nn modules function as model (can do a forward pass)
















