class EarlyStopping:
    """
    EarlyStopping class to stop training when a monitored metric has stopped improving.
    Attributes:
        patience (int): Number of epochs to wait after the last time the monitored metric improved.
        min_delta (float): Minimum change in the monitored metric to qualify as an improvement.
        best_loss (float or None): The best recorded value of the monitored metric.
        counter (int): Counts the number of epochs since the last improvement.
        should_stop (bool): Indicates whether training should be stopped.
    Methods:
        __init__(patience=3, min_delta=0.0):
            Initializes the EarlyStopping instance with the given patience and min_delta values.
        step(val_loss):
            Updates the state of the EarlyStopping instance based on the validation loss.
            Args:
                val_loss (float): The current value of the monitored metric.
            Returns:
                bool: True if training should be stopped, False otherwise.
    """
    
    def __init__(self, patience=3, min_delta=0.0):

        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0 
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop

