import torch
from torch import nn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

print(torch.__version__)
print("make device agnostic code")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device is :{device}")

# Make 1000 samples
n_samples = 1000

# Create circles
X, y = make_circles(n_samples, noise=0.03)
print(" Create circles")
print(f"First 5 X features:\n{X[:5]}")
print(f"\nFirst 5 y labels:\n{y[:5]}")


print(" turn data into tensors")
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)
print(f"First 5 X features:\n{X[:5]}")
print(f"\nFirst 5 y labels:\n{y[:5]}")


print(" split data for train and test")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train = X_train.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)
print(f"len of X_train:{len(X_train)}  | len of X_test:{len(X_test)}")
print(f"len of y_train:{len(y_train)}  : len of y_test:{len(y_test)}")


class CircleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU()#activation function

    def forward(self, x):
        # intersperse the ReLU activation function between layers
        return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))

print("instance a circlemodel")
my_model = CircleModel().to(device)
print(f"my_model is:{my_model}")

print("setup loss function and optimizer")
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(my_model.parameters(), lr=0.1)


print("start training")
epochs = 10000

for epoch in range(epochs):
    ### train
    my_model.train()
    # 1. Forward pass
    y_logits = my_model(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits)) # logits -> prediction probabilities -> prediction labels

    # 2. Calculate loss and accuracy
    loss = loss_fn(y_logits, y_train)
    #acc = accuracy_fn(y_true=y_train, y_pred=y_pred)

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backward
    loss.backward()

    # 5. Optimizer step
    optimizer.step()

    #### Test
    my_model.eval()
    with torch.inference_mode():
        # 1. Forward pass
        test_logits = my_model(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))

        # 2. Calculate loss and accuracy
        test_loss = loss_fn(test_logits, y_test)
        #test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)
    
    # print out what's happening
    if epoch % 10 == 0:
        print(f"Epoch:{epoch} | Loss:{loss:.5f}")




    