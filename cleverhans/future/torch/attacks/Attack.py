"""
Base Class for PyTorch Attacks
"""

class Attack():
    """
    This class is the base class for all the PyTorch Attacks

    Args:
        model: The model nn.Module object
        dtype: A string mentioning the data type of the model
    """
    def __init__(self, model:nn.Module, dtype:str):
        self.model = model
        self.dtype = dtype
