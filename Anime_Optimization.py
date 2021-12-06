# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 12:16:52 2021

@author: Gary
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

st.title("Anime Optimization")
st.markdown("https://github.com/GaryDLang/AnimeOptimization")
st.markdown("Based on the anime.csv dataset: https://www.kaggle.com/CooperUnion/anime-recommendations-database")
st.write("The goal of this data analysis is to find the optimal genre, format and length for yielding the highest audience approval and interaction.")
## Data Uploading and Cleaning
df = pd.read_csv('anime.csv')

## Removing nan and Unknown values
df = df[df.notna().all(axis = 1)]
    ## there are 187 values in the "episodes" column with "Unknown"
df = df[~((df["episodes"] == "Unknown") & ((df["rating"] <= 6.0) | (df["members"] <= 5000)))]
    ## updates the remaining entries vital to the research with accurate data
eps_fix = {74:981,252:988,615:500,991:1077,1021:7,1272:148,1930:131,1993:30,2530:43,2810:140,3055:24,3574:49,4735:26,5500:38}
for x in eps_fix:
    df.loc[x,"episodes"] = eps_fix[x]
## converts to numeric
def can_be_numeric(c):
    try:
        pd.to_numeric(df[c])
        return True
    except:
        return False
    
num_cols = [c for c in df.columns if can_be_numeric(c)]
df[num_cols] = df[num_cols].apply(pd.to_numeric, axis = 0)
num_cols.remove("anime_id") ## whlie numeric, "anime_id" is not a factor
## changes genre column to list of strings
df["genre"] = df["genre"].map(lambda x: x.split(", "))
## unique lists of genres and type
types = set()
df["type"].map(types.add)
types = sorted(list(types))
    
df["Long_Running"] = df["episodes"] > 36


### Types Analysis ###
    
    #-- Creation of types_stats --#
col_d = "Average_Members"
col_e = "Average_Rating"
type_mem = [df[df["type"]==i]["members"].mean() for i in types]
type_av = [df[df["type"]==i]["rating"].mean() for i in types]
types_stats = pd.DataFrame(list(zip(types,type_mem,type_av)), columns =["Type", col_d, col_e])
    
    #-- Prompts for chart --#
st.write("To first begin, the five different video formats must be analyzed.")
type_choice = st.selectbox("Select which popularity metric to graph.", [col_d,col_e])
    
    #-- TypeChart Initialization --#
type_chart = alt.Chart(types_stats).mark_bar().encode(
x = "Type",
y = type_choice,
color = alt.value("red")
).properties(
width = 700,
height = 400)
        
st.altair_chart(type_chart)
    
st.write("Clearly, the TV format is most ideal for maintaining a larger fanbase, as it has the highest average membership by a wide margin. Moving forward, the analysis will focus on TV based anime.")
    
## shrinking data set to only TV shows
df = df[df["type"] == "TV"]
    
    
### Genres Analysis ###
    
    # sorted list of genres
genres = sorted(list(set(df["genre"].sum())))
    
    # Prompt #
sel_gen = st.selectbox("Select a Genre to Optimize",genres)
    
    #-- Creation of genre_stats --#
col_f = f"Avg_Rating_Sub_{sel_gen}"
col_g = f"Portion of {sel_gen} Anime"
## Renders Avg Rating of sel_gen
av = [df[df["genre"].map(lambda g_list: i in g_list)]["rating"].mean() for i in genres]
## Length of subset
sub_tot = sum(df["genre"].map(lambda g_list: sel_gen in g_list))
## renders proportion of each genre in the sel_gen DataFrame
perc = [(round(sum(df["genre"].map(lambda g_list: (sel_gen in g_list) & (i in g_list))/sub_tot),4)) for i in genres]
## Renders Avg Rating per genre
sub_av = [df[df["genre"].map(lambda g_list: (sel_gen in g_list) & (i in g_list))]["rating"].mean() for i in genres]
## Constructs dataframe
genre_stats = pd.DataFrame(list(zip(genres,av,perc,sub_av)), columns =["Genre", "Average_Rating", col_g, col_f])
st.markdown("Source: https://www.geeksforgeeks.org/create-a-pandas-dataframe-from-lists/")
    
## cleaner way of yielding and displaying average rating
temp = round(genre_stats[genre_stats["Genre"] == sel_gen].iloc[0,3],2)
st.write(f"The {sel_gen} genre as a whole has an average user rating of {temp} out of 10.")
    
    
sub_genre_chart = alt.Chart(genre_stats[genre_stats['Genre'] != sel_gen]).mark_point().encode(
alt.X(col_f, scale=alt.Scale(zero=False)),
y = col_g,
color = alt.value("orange"),
shape = alt.value("diamond"),
tooltip = ['Genre',col_g,col_f]
).properties(
width = 700,
height = 400)
    
## horizontal line for threshhold visualization
line = alt.Chart(pd.DataFrame({'y': [.1]})).mark_rule().encode(y='y')
st.altair_chart(sub_genre_chart + line)
st.markdown("Source: https://github.com/altair-viz/altair/issues/2059")
    
## find overall ideal genre
ovr_max = max(genre_stats["Average_Rating"])
ovr_max = genre_stats[genre_stats["Average_Rating"] == ovr_max].iloc[0,0]
    
    
## Selects for genre within threshhold with highest rating
gen_max = max(genre_stats[genre_stats[col_g] > 0.1][col_f])
gen_max = genre_stats[genre_stats[col_f] == gen_max].iloc[0,0]
st.markdown(f"To avoid heavy data bias, the ideal genre is that with the highest average rating above the threshhold of ten percent of the parent genre. In this case, the ideal secondary genre for {sel_gen} is {gen_max}. (Hover over for details.)")
    
    
st.write(f"The genre with the overall highest average rating is {ovr_max}.")
    
def highlight_max(s, props=''):
    return np.where(s == np.nanmax(s.values), props, '')
def subpar(s, props=''):
    return np.where(s < 7.0, props, '')
slice_ = 'Average_Rating'
gen_color = genre_stats.style.apply(highlight_max, props='color:green;', axis=0, subset=slice_)\
            .apply(subpar, props='color:red;', axis=0, subset=slice_)
gen_color

st.markdown("Source: https://pandas.pydata.org/pandas-docs/stable/user_guide/style.html#Styler-Functions")

### Episodes Analysis ###
    
col_a = "members"
col_b = "rating"
col_c = "Long_Running"
    
df[col_c] = df["episodes"] > 36
    
## applies standard scaling, then grandfathers in classification column
scaler = StandardScaler()
scaler.fit(df[num_cols])
dfd = pd.DataFrame(scaler.transform(df[num_cols]), columns = num_cols)
dfd[col_c] = df[col_c]
    
## cleans clf column of NaN values since notna didn't
dfd = dfd[~((dfd[col_c] != True) & (dfd[col_c] != False))]
## fixes "type" error with KNClassifier
dfd[col_c] = dfd[col_c].map(lambda x: "Short" if x == False else "Long")
    
    
st.markdown(f"Next is to decide upon how long our {sel_gen}, {gen_max} TV series should be.")
    
## to accomodate Altair 5000 entry limitation, changes starting value, thus whole set
dfd_slice = st.selectbox("To accomodate Altair data set is divided into three slices uniformly. Pick one of three slices.", [1,2,3])
A, _ = dfd.shape
dfd = dfd.iloc[dfd_slice:A:3,:]
    
    
chart_true = alt.Chart(dfd).mark_circle().encode(
x = alt.X(col_a,scale=alt.Scale(zero=False)),
y = alt.Y(col_b,scale=alt.Scale(zero=False)),
color=col_c
).properties(
width = 700,
height = 400)
st.markdown("Source: https://altair-viz.github.io/user_guide/customization.html")
        
st.altair_chart(chart_true)
    
    
    ### Machine Learning ###
st.markdown("Using the KNeighborsClassifier algorithm, we can test if the popularity metrics trend toward longevity or brevity.")
inslider = st.slider("Select a number of iterations to compute.", 1, 50)
    
clf = KNeighborsClassifier(n_neighbors = inslider)
clf.fit(dfd[[col_a,col_b]],dfd[col_c])
dfd["Prediction"] = clf.predict(dfd[[col_a,col_b]])
    
chart_neighbors = alt.Chart(dfd).mark_circle().encode(
x = alt.X(col_a,scale=alt.Scale(zero=False)),
y = alt.Y(col_b,scale=alt.Scale(zero=False)),
color='Prediction'
).properties(
width = 700,
height = 400)
    
st.altair_chart(chart_neighbors)
    
st.markdown("Regardless of the slice or the number of iterations, the data never reaches to distinct groups. There does not appear to be a direct correlation between longevity and involvement.")
    
    
