import numpy as np
import pandas as pd
from pykalman import KalmanFilter
from dataclasses import dataclass


@dataclass
class featureProcessor:
    X: str
    y: str

    def __init__(self, X, y):
        self.X = X
        self.y = y
        return
