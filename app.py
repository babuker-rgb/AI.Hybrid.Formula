# --- Minimal code that should run on Streamlit Cloud ---
import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import plotly.express as px

st.set_page_config(page_title="Minimal PINN")
st.title("Minimal PINN – test")

# Dummy model to test
class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 3)
    def forward(self, x):
        return self.fc(x)

model = DummyModel()
st.success("Model loaded")

# Generate dummy data
X = np.random.randn(100, 8)
y = np.random.randn(100, 3)
df = pd.DataFrame(X, columns=['a','b','c','d','e','f','g','h'])
st.dataframe(df.head())

st.write("If you see this, PyTorch and Plotly work.")
