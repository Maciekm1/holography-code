#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 22:04:09 2022

@author: erick
"""

""" Tracking App"""
#%%
from email.policy import default
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


st.title("Make Tracks")

#########################################
#          Import Data
#########################################

st.header("Import Data Set as CSV File")

data = st.file_uploader("")
DF = pd.read_csv(data)

st.dataframe(DF)


#########################################
#          Tracking
#########################################

st.header("Choose Tracking Algorithm")

# @st.cache(suppress_st_warning=True)
def choose_algorithm():

    algorithm = st.radio("Algorithm",
                        ("Search Sphere", "DBSCAN", "KMeans"))

    if algorithm == "Search Sphere":
        
        rsphere = st.number_input("Sphere Radius", value=5)
        frame_skip = st.number_input("Frame Skip", value=10)
        min_size = st.number_input("Track Minimum Size", value=50)
        args = [rsphere, frame_skip, min_size]
        
            
    elif algorithm == "DBSCAN":
        
        cores = os.cpu_count()
        eps = st.number_input("Epsilon (e.g. 5)", value=5)
        min_samples = st.number_input("Track Minimum Size", value=50)
        args = [cores, eps, min_samples]
        
    elif algorithm == "KMeans":
        
        args = []
        
    return algorithm, args

algorithm, args = choose_algorithm()

@st.cache(suppress_st_warning=True)
def tracking_fun(DF, algorithm, args):
    # st.header("Choose Tracking Algorithm")


    # algorithm = st.radio("Algorithm",
    #                     ("Search Sphere", "DBSCAN", "KMeans"))

    if algorithm == "Search Sphere":
        
        # rsphere = st.number_input("Sphere Radius", value=5)
        # frame_skip = st.number_input("Frame Skip", value=10)
        # min_size = st.number_input("Track Minimum Size", value=50)
        
        rsphere = args[0]
        frame_skip = args[1]
        min_size = args[2]
        
        # start = st.button("Start Tracking")
        # if start:
        T0 = time.time()
        LINKED = f.search_sphere_tracking(DF, float(rsphere), float(frame_skip), float(min_size))
        st.dataframe(LINKED)
        T = time.time()
        st.success("Runtime was "+str(np.float16(T-T0))+" seconds")
            
    elif algorithm == "DBSCAN":
        
        
        # cores = os.cpu_count()
        # eps = st.number_input("Epsilon (e.g. 5)", value=5)
        # min_samples = st.number_input("Track Minimum Size", value=50)
        
        cores = args[0]
        eps = args[1]
        min_samples = args[2]
        
        # start = st.button("Start Tracking")
        # if start:
        T0 = time.time()
        DBSCAN = cl.DBSCAN(eps=float(eps), min_samples=int(min_samples), n_jobs=cores).fit(DF[['X', 'Y', 'Z']])
        DF['PARTICLE'] = DBSCAN.labels_
        LINKED = DF
        LINKED = LINKED.drop(np.where(LINKED.PARTICLE.values == -1)[0])
        # st.dataframe(LINKED)
        T = time.time()
        st.success("Runtime was "+str(np.float16(T-T0))+" seconds")
        
    elif algorithm == "KMeans":
        
        # start = st.button("Start Tracking")
        # if start:
        D = DF.values
        T0 = time.time()
        kmeans = cl.KMeans(n_clusters=D[D[:, 5] == 0].shape[0], init='k-means++').fit(DF[['X', 'Y', 'Z']])
        DF['PARTICLE'] = kmeans.labels_
        LINKED = DF
        # st.dataframe(LINKED)
        T = time.time()
        st.success("Tracking runtime was "+str(np.float16(T-T0))+" seconds")
    
    return LINKED

with st.form(key="form"):
    # start = st.button("Start Tracking")
    # if start:
        # LINKED = tracking_fun(DF, algorithm, args)
    submit_button = st.form_submit_button(label="Start Tracking")


LINKED = tracking_fun(DF, algorithm, args)

@st.cache
def convert_df(df):
   return df.to_csv().encode('utf-8')

csv_linked = convert_df(LINKED)

st.download_button(
   "Press to Download Linked Tracks",
   csv_linked,
   "file.csv",
   "text/csv",
   key='download-csv'
)


#########################################
#          Smoothing
#########################################

st.header("Track Smoothing")

# @st.cache(suppress_st_warning=True)
def smoothing_fun(LINKED):

    spline_degree = st.number_input("Spline Degree", value=3)
    sc = st.number_input("Smoothing Condition", value=0.999, format="%f")
    filter_data = st.checkbox("Fitler Data For Jumps?", value=True)

    if filter_data:
        limit = st.number_input("Filter Data Limit", value=5)
    else: 
        limit = 5
    
    smooth = st.button("Start Smoothing")
    if smooth:
        particle_num = np.sort(LINKED.PARTICLE.unique())
        T0_smooth = time.time()
        smoothed_curves = -np.ones((1, 5))

        for pn in particle_num:
            # Do not use this
            # L = LINKED[LINKED.PARTICLE == pn].values
            # X = f.smooth_curve(L, spline_degree=spline_degree, lim=20, sc=3000)
            
            L = LINKED[LINKED.PARTICLE == pn]
            # temp = f.clean_tracks(L)
            # temp = f.clean_tracks_search_sphere(L, 5)
            # L = pd.DataFrame.transpose(pd.DataFrame(temp, ['X', 'Y', 'Z', 'TIME', 'FRAME','PARTICLE']))

            if len(L) < 100:
                continue
            X = f.csaps_smoothing(L, smoothing_condition=sc, filter_data=filter_data, limit=limit)
            
            if X != -1:
                smoothed_curves = np.vstack((smoothed_curves, np.stack((X[0], X[1], X[2], X[3], pn*np.ones_like(X[1])), axis=1))) 

        smoothed_curves = smoothed_curves[1:, :]
        smoothed_curves_df = pd.DataFrame(smoothed_curves, columns=['X', 'Y' ,'Z', 'TIME','PARTICLE'])
        T_smooth = time.time() - T0_smooth
        # st.dataframe(smoothed_curves_df)

        st.success("Smoothing runtime was "+str(np.float16(T_smooth))+" seconds")
        
    return smoothed_curves_df

smoothed_curves_df = smoothing_fun(LINKED)


# show_plot = st.checkbox("Show Plot", value=True)
# if show_plot:
fig = px.line_3d(smoothed_curves_df, x='X', y='Y', z='Z', color='PARTICLE', hover_data=['TIME'])
fig.update_traces(marker=dict(size=1))
st.plotly_chart(fig, use_container_width=True)

csv_smoothed = convert_df(smoothed_curves_df)

st.download_button(
   "Press to Download_Smoothed Curves",
   csv_smoothed,
   "file.csv",
   "text/csv",
   key='download-csv'
)

#########################################
#          Analysis
#########################################

st.header("Analysis")

particle_num = np.unique(smoothed_curves_df['PARTICLE'])

xx, yy, zz, tt, pp, sp = -1, -1, -1, -1, -1, -1

for pn in particle_num:
    s = smoothed_curves_df[smoothed_curves_df['PARTICLE'] == pn]
    # print(pn, len(s))

    if len(s) > 100:
        speed, x, y, z, t = f.get_speed(s)
        xx = np.hstack((xx, x))
        yy = np.hstack((yy, y))
        zz = np.hstack((zz, z))
        tt = np.hstack((tt, t))
        pp = np.hstack((pp, pn*np.ones(len(t))))
        sp = np.hstack((sp, speed))
    

tracks_w_speed = pd.DataFrame(np.transpose([xx[1:], yy[1:], zz[1:], tt[1:], pp[1:], sp[1:]]), columns=['X', 'Y', 'Z', 'TIME', 'PARTICLE', 'SPEED'])
mean_speed = np.mean(tracks_w_speed.SPEED)
st.dataframe(tracks_w_speed)

csv_speed = convert_df(tracks_w_speed)

st.download_button(
   "Press to Download File with Speed",
   csv_speed,
   "file.csv",
   "text/csv",
   key='download-csv'
)

fig = px.histogram(tracks_w_speed, x='SPEED', title='Bacteria Speed Frequency ' + '&mu;<sub>s</sub> = ' + str(np.float16(mean_speed))+' &mu;m/s')
# fig.update_traces(marker=dict(size=1))
fig.add_vline(x=mean_speed, line_dash = 'dash', line_color = 'firebrick')

st.plotly_chart(fig, use_container_width=True)