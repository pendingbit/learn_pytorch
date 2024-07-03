# Import PyTorch and matplotlib
import torch
from torch import nn #nn contains all of PyTorch's building blocks for neural networks
#import mutplotlib.pyplot as plt
from pathlib import Path

#Check PyTorch version
print(torch.__version__)


#Setup device agnostic code 
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device is {device}")

#Create Data
#Create weight and bias
weight = 0.6
bias = 0.2

#Create range values
start = 0
end = 1
step = 0.02

#Create X and y (features and labels)
X = torch.arange(start, end, step).unsqueeze(dim=1) #without unsqueeze, errors will happen later on (shapes within linear layers)
y = weight * X + bias

#Split data
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

X_train = X_train.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)

# Subclass nn.Module to make our model
class LinearRegressionModle(nn.Module):
    def __init__(self):
        super().__init__()
        #Use nn.Linear() for createing the model parameters
        self.linear_layer = nn.Linear(in_features=1, out_features=1)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)

model_1 = LinearRegressionModle()
print(f"Current model in {next(model_1.parameters()).device}")
print("Move model to cuda")
model_1.to(device)
print(f"Current model in {next(model_1.parameters()).device}")

#Create loss function
loss_fn = nn.L1Loss()

#Ceate optimizer
optimizer = torch.optim.SGD(params=model_1.parameters(), lr=0.01)

epochs = 1000


print(f"Inital parameters: {list(model_1.parameters())}")

for epoch in range(epochs):
    ### Training
    model_1.train() # default construction to start train model

    # 1. Forward pass
    y_pred = model_1(X_train)

    # 2. Calculate loss
    loss = loss_fn(y_pred, y_train)

    # 3. Zero grad optimizer
    optimizer.zero_grad()

    # 4. Loss backward
    loss.backward()

    # 5.Setup the optimizer
    optimizer.step()

    ### Testing
    model_1.eval() #default construction to start evaluate model

    with torch.inference_mode():
        # 1. Forward pass
        test_pred = model_1(X_test)

        # 2. Calculate the loss
        test_loss = loss_fn(test_pred, y_test)
    
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Train loss: {loss} | Test loss: {test_loss}")


print(f"Trained parameters: {list(model_1.parameters())}")
print(f"Target parameters: {weight} {bias}")



###################################SAVE MODEL#########################################
# 1. Create models directory
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. Create model save path
MODEL_NAME = "01_PyTorch_Workflow_Model.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Save the model state dict
print(f"Saving model to : {MODEL_SAVE_PATH}")
torch.save(obj=model_1.state_dict(), f=MODEL_SAVE_PATH) #only save the models learned parameters




###################################LOAD MODEL#########################################
# 1. Instantiate a fresh instance of LinearRegressModle
loaded_model_1 = LinearRegressionModle()

# 2. Load model state dict
loaded_model_1.load_state_dict(torch.load(MODEL_SAVE_PATH))

# 3. Put model to target device (if your data is on GPU, model will have to be on GPU to make predictions)
loaded_model_1.to(device)

print(f"Loaded model:\n{loaded_model_1}")
print(f"Model on device:\n{next(loaded_model_1.parameters()).device}")



##########################PREDICT With LOADED MODEL###################################
# Evaluate loaded model
loaded_model_1.eval()
with torch.inference_mode():
    loaded_model_1_pred = loaded_model_1(X_test)

print(test_pred == loaded_model_1_pred)
print("\n\nHELLO PYTORCH")
