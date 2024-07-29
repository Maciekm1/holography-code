import streamlit as st
import os
import math as m
import time
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import easygui
import matplotlib.pyplot as plt
import functions as f
import sklearn.cluster as cl
import hdbscan
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import plotly.express as px
import pandas as pd
import easygui as gui
from plotly.offline import plot
from bokeh.plotting import figure
import seaborn as sns
import plotly.figure_factory as ff

st.header("Import Data Set as CSV File")

st.cache()
def import_data():
    data = st.file_uploader("/media/erick/NuevoVol/LINUX_LAP/PhD/Test")
    return pd.read_csv(data)

DF = import_data()
st.dataframe(DF)