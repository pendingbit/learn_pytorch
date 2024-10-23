#import dependence
import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from torch import nn

#set the hyperparameters for data creation
NUM_CLASSES = 4
NUM_FEATURES = 2

# 1. create multi-class data
X_blob, y_blob = make_blobs(n_samples=1000,n_features=NUM_FEATURES,centers=NUM_CLASSES,cluster_std=1.5)

# 2. turn data into tensor
X_blob = torch.from_numpy(X_blob).type(torch.float)
y_blob = torch.from_numpy(y_blob).type(torch.LongTensor)
print(f"first 5 X_blob is {X_blob[:5]}")
print(f"first 5 y_blob is {y_blob[:5]}")

# 3. split int train and test sets
X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(X_blob, y_blob, test_size=0.2)

# 4. plot data
plt.figure(figsize=(10, 7))
plt.scatter(X_blob[:,0], X_blob[:, 1], c=y_blob, cmap=plt.cm.RdYlBu)

device = "cuda" if torch.cuda.is_available() else "cpu"

# build model
class BlobModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=8):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_features)
        )

    def forward(self, x):
        return self.linear_layer_stack(x)
        

model = BlobModel(input_features=NUM_FEATURES, output_features=NUM_CLASSES, hidden_units=8).to(device)


# create loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)


epochs = 10000

X_blob_train, y_blob_train = X_blob_train.to(device), y_blob_train.to(device)
X_blob_test, y_blob_test = X_blob_test.to(device), y_blob_test.to(device)

for epoch in range(epochs):
    ### Train
    model.train()

    # 1. forward pass
    y_logits = model(X_blob_train) # model outputs raw logits
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1) #logits -> prediction probablities -> prediction labels

    # 2. calculate loss and accuracy
    loss = loss_fn(y_logits, y_blob_train)
    #acc = accuracy_fn(y_true=y_blob_train, y_pred=y_pred)

    # 3. optimizer zero grad
    optimizer.zero_grad()

    # 4. loss backwards
    loss.backward()

    # 5. optimizer setp
    optimizer.step()

    ### Test
    model.eval()
    with torch.inference_mode():
        # 1. forward pass
        test_logits = model(X_blob_test)
        test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)

        # 2. calculate loss
        test_loss = loss_fn(test_logits, y_blob_test)
        #test_acc = accuracy_fn(y_true=y_blob_test, y_pred=test_pred)


    if epoch % 10 == 0:
        print(f"Epoch:{epoch} | Loss: {loss:.5f} ")





