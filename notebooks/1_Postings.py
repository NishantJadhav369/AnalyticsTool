import time  # to simulate a real time data, time loop
import numpy as np  # np mean, np random
import pandas as pd  # read csv, df manipulation
import plotly.express as px  # interactive charts
import streamlit as st  # 🎈 data web app development
from datetime import datetime as dt
from dateutil.relativedelta import relativedelta
from services.utils import compute_posts_growth


# read csv from a github repo
dataset_link = '../data/jobs_data.csv'

# read csv from a URL
def get_data() -> pd.DataFrame:
    return pd.read_csv(dataset_link)

df = get_data()

# dashboard title
st.title("Data Trend Navigator Dashboard")

df['date_posted'] = df.date_posted.apply(lambda x:dt.strptime(x,"%m/%d/%Y"))
max_date = df.date_posted.max() 
min_date = df.date_posted.min() 
curr_start_val = max_date - relativedelta(days =31)  
curr_end_val =max_date


# creating a single-element container
placeholder = st.empty()


with placeholder.container():
    
    filter1, filter2, filter3, kpi1 = st.columns(4)
    
    with filter1:
        vertical = st.selectbox("Vertical", ['company_name','blend_industry'])
        
    with filter2:
        start_date = st.date_input("Start Date", min_value=min_date, max_value=max_date,value=curr_start_val,format="MM/DD/YYYY")
        start_date = str(start_date).split()[0]
        curr_start = dt.strptime(start_date,"%Y-%m-%d") 

    with filter3:
        end_date = st.date_input("End Date", min_value=min_date, max_value=max_date,value=curr_end_val,format="MM/DD/YYYY")
        end_date = str(end_date).split()[0]
        curr_end =  dt.strptime(end_date,"%Y-%m-%d") 
        
    interval = curr_end - curr_start
    prev_end = curr_end - interval
    prev_start = curr_start - interval
    
    # fill in those three columns with respective metrics or KPIs
    kpi1.metric(
        label="Interval Period (Days)",
        value=interval.days
    )
    
    st.markdown("### Postings Growth within interval period.")
    tab1, tab2, tab3, tab4 = st.tabs(["All","Data Scientist", "Data Engineer", "Data Analyst"])
    
    
    df2 = df.copy()
    
    compute_posts_growth(df2,vertical,curr_start,curr_end,prev_start,prev_end)
    compute_posts_growth(df2,vertical,curr_start,curr_end,prev_start,prev_end,tab_value='Data Scientist',path='ds')
    compute_posts_growth(df2,vertical,curr_start,curr_end,prev_start,prev_end,tab_value='Data Engineer',path='de')
    compute_posts_growth(df2,vertical,curr_start,curr_end,prev_start,prev_end,tab_value='Data Analyst',path='da')
   
    
    with tab1:
        top_all = pd.read_csv('services/post_growth_all.csv')
        top_all.index =top_all.index +1
        fig1 = px.bar(top_all,vertical,'growth_pct')
        st.write(fig1)
        
        st.markdown("### Detailed Data View")
        st.dataframe(top_all)
    
    with tab2:
        top_ds = pd.read_csv('services/post_growth_ds.csv')
        top_ds.index =top_ds.index +1
        fig2 = px.bar(top_ds,vertical,'growth_pct')
        st.write(fig2)
        
        st.markdown("### Detailed Data View")
        st.dataframe(top_ds)
    
    with tab3:
        top_de = pd.read_csv('services/post_growth_de.csv')
        top_de.index =top_de.index +1
        fig3 = px.bar(top_de,vertical,'growth_pct')
        st.write(fig3)
        
        st.markdown("### Detailed Data View")
        st.dataframe(top_de)
        
    with tab4:
        top_da = pd.read_csv('services/post_growth_da.csv')
        top_da.index =top_da.index +1
        
        fig4 = px.bar(top_da,vertical,'growth_pct')
        st.write(fig4)
        
        st.markdown("### Detailed Data View")
        st.dataframe(top_da)
