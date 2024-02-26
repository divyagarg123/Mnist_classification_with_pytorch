# Mnist_classification_with_pytorch

## This module has three files :
* **utils.py**: All the utility functions to be used.
* **model.py**: Contains the model architecture.
* **S5.ipynb**: Entry point for training and testing

### utils.py:
  * get_device(): Get the device (CPU or cuda).
  * print_summary(model): print the summary of model.
  * plot_images(batch_data, batch_label): plot images in training data for visualization.
  * get_optimizer(model): Get the optimizer.
  * loss_function(): Get the loss function .
  * scheduler(optimizer): Get the scheduler.
  * get_train_test_data(): Load train test data after applying transformations.
  * GetCorrectPredCount(pPrediction, pLabels): Check how many samples were predicted correct in test data.
  * train(model, device, train_loader, optimizer, criterion): Train the model using train dataset.
  * test(model, device, test_loader, criterion): Test the model using test dataset.
  * plot_loss_acc_curves(train_losses, train_acc, test_losses, test_acc): Plot training and test curves.

### model.py:
  * class Net(nn.Module):
    * __init__(self): Initialize the model with all layers defined.
    * forward(self, x): Make the architecture of model.
   
### S5.ipynb:
* Call the above apis to train and test the model for given no of epochs.

## How to Run the script:
* Clone the repo in jupyter-notebook in colab or local.
* Run S5.ipynb.

