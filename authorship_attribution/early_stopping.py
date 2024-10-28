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
                validation_loss (float): The current value of the monitored metric.
            Returns:
                bool: True if training should be stopped, False otherwise.
    """
    
    def __init__(self, patience=3, min_delta=0.0):

        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def step(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

