# Linear Regressor.py
import torch
from torch import nn
import math
import matplotlib.pyplot as plt

class LinearRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.randn(1, dtype=torch.float), 
                                   requires_grad=True) #

        self.bias = torch.nn.Parameter(torch.randn(1, dtype=torch.float), 
                                requires_grad=True) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias  
     
def randomised_data(data= None):
    weight = 0.3
    bias = 0.7
    
    if data is None:
        x = torch.arange(0,10,0.1)
        random_indices = torch.randperm(x.size(0))
    
        X = x[random_indices]
        Y = weight * X + bias
    else:
        x,y = data
        
        random_indices = torch.randperm(x.size(0))
        X = x[random_indices]
        Y = y[random_indices]


    
    return X,Y

def graph_data(train_valx, train_val_y, test_val_x, test_val_y, predictions= None):
    buffer = 5
    max_x = max(train_valx+buffer)
    max_y = max(train_val_y+buffer)
    
    plt.figure(figsize=(max_x,max_y))
      
    plt.scatter(train_valx, train_val_y, c="b", s=4, label="Training data")
  
    # Plot test data in green
    plt.scatter(test_val_x, test_val_y, c="g", s=4, label="Testing data")
    
    if predictions is not None:
        plt.scatter(test_val_x, predictions, c="r", s=4, label="Model Predictions")
        
      # Show the legend
    plt.legend(prop={"size": 14});
 
def train_linear_regressor(data):
    model = LinearRegressor()
    
    torch.manual_seed(123)
    x,y = randomised_data()
    
    #80:20 train test split
    train_size = math.floor(x.size(0)*0.8)
    
    X_train = x[:train_size]
    Y_train = y[:train_size]
    X_test = x[train_size:]
    Y_test = y[train_size:]
    
    
    model = LinearRegressor()
    '''-----------------
    TRAINING LOOP    '''
    
    loss_fn = torch.nn.MSELoss()
    optimiser = torch.optim.SGD(params=model.parameters(), lr= 0.01)
    
    epochs = 500
    
    epoch_count = []
    train_loss_values = []
    test_loss_values = []
    
    for epoch in range(epochs):
        model.train()
        y_pred = model(X_train)
    
        
        loss = loss_fn(y_pred,Y_train)
        
        optimiser.zero_grad()
        
        loss.backward()
        
        optimiser.step()
        # Testing code
        #model in evaluation mode
        model.eval()
        
        #inference mode
        with torch.inference_mode():
            #forward_pass
            test_pred = model(X_test)
            # calc loss
            test_loss = loss_fn(test_pred, Y_test)
        
        #
        if epoch % 10 == 0:
            epoch_count.append(epoch)
            train_loss_values.append(loss)
            test_loss_values.append(test_loss)
            print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss} ")
        
    train_loss_numpy = [loss.detach().cpu().numpy() for loss in train_loss_values]
    test_loss_numpy = [loss.detach().cpu().numpy() for loss in test_loss_values]
    return model

     
    

def main():
    torch.manual_seed(123)
    x,y = randomised_data()
    
    #80:20 train test split
    train_size = math.floor(x.size(0)*0.8)
    
    X_train = x[:train_size]
    Y_train = y[:train_size]
    X_test = x[train_size:]
    Y_test = y[train_size:]
    
    
    model = LinearRegressor()
    '''-----------------
    TRAINING LOOP    '''
    
    loss_fn = torch.nn.MSELoss()
    optimiser = torch.optim.SGD(params=model.parameters(), lr= 0.01)
    
    epochs = 500
    
    epoch_count = []
    train_loss_values = []
    test_loss_values = []
    
    height_of_graph = 10
    
    for epoch in range(epochs):
        model.train()
        y_pred = model(X_train)
    
        
        loss = loss_fn(y_pred,Y_train)
        
        optimiser.zero_grad()
        
        loss.backward()
        
        optimiser.step()
        # Testing code
        #model in evaluation mode
        model.eval()
        
        #inference mode
        with torch.inference_mode():
            #forward_pass
            test_pred = model(X_test)
            # calc loss
            test_loss = loss_fn(test_pred, Y_test)
        
        #
        if epoch % 10 == 0:
            epoch_count.append(epoch)
            train_loss_values.append(loss)
            test_loss_values.append(test_loss)
            print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss} ")
        
    train_loss_numpy = [loss.detach().cpu().numpy() for loss in train_loss_values]
    test_loss_numpy = [loss.detach().cpu().numpy() for loss in test_loss_values]

    # Plot the loss curves using the clean numpy arrays
    plt.plot(epoch_count, train_loss_numpy, label="Train loss")
    plt.plot(epoch_count, test_loss_numpy, label="Test loss")
    plt.title("Training and test loss curves")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()
    


if __name__ == "__main__":
    main() 
