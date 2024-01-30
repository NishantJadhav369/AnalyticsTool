## Code for Helpers function


import streamlit as st 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import catboost
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

@st.cache_data
def load_data(path1,path2,path3,path4):
    df1 = pd.read_excel(path1,dtype={'PHASE':str})
    df2 = pd.read_excel(path2,dtype={'PHASE':str})
    df3 = pd.read_excel(path3)
    df4 = pd.read_csv(path4)
    
    df1['ID_JOB'] = df1['SHORTAGE'] + df1['ORIGINAL_JOB_NUM'] + df1['REASON']
    df4['Claim Component'] =  df4['Claim Component'].str.strip('[]')
    df1 = pd.merge(df1,df4.loc[:, df4.columns != 'REASON'],'left',['ID_JOB']) # merge with regression data
    df1 = df1.fillna('missing')
    
    return df1,df2,df3,df4

@st.cache_data
def eda_table(eda_slice,text_component, ship, filter_dim, filter_val, group_level,level,time_agg_type):
    
    if filter_dim!='None':
        eda_slice = eda_slice[eda_slice[filter_dim] == filter_val] # EDA slice 1
        
    if group_level in ['Sales Region ID','Sales District ID']:
        ship_rd = ship[['Shortage','Reason','Original Job Number','Sales Region ID','Sales District ID']].drop_duplicates()
        eda_slice = pd.merge(eda_slice,ship_rd,'left',['Shortage','Reason','Original Job Number'])
        
    if group_level =='Piecemark':
        ship_piecemark = ship.groupby(['Shortage','Original Job Number','Reason','Piecemark'])['Quantity'].sum().reset_index()
        ship_piecemark = ship_piecemark[['Shortage','Original Job Number','Reason','Piecemark','Quantity']].drop_duplicates()
        piecemark_quantity = pd.merge(eda_slice,ship_piecemark,'left',['Shortage','Reason','Original Job Number'])
        agg = piecemark_quantity.groupby(['Piecemark'])['Quantity'].sum().sort_values(ascending=False).reset_index()[:10]
        return agg,eda_slice
    
    if group_level in ['GL Period','Release Date']:       
        
        
        
        
        agg = eda_slice.drop_duplicates([group_level,'Shortage','Original Job Number','Reason','Shortage Cost Spread'])
        agg = agg.groupby(['Shortage','Original Job Number','Reason',group_level])['Shortage Cost Spread'].sum().reset_index()
        agg = agg.groupby([group_level])['Shortage Cost Spread'].sum().reset_index()
        if time_agg_type  == 'QoQ':
            agg = agg.set_index(group_level).resample('Q').sum()
            agg = agg.reset_index()
        elif time_agg_type  == 'YoY':
            agg = agg.set_index(group_level).resample('Y').sum()
            agg = agg.reset_index()
        elif time_agg_type  == 'WoW':
            agg = agg.set_index(group_level).resample('W').sum()
            agg = agg.reset_index()
        else: 
            agg = agg.set_index(group_level).resample('M').sum()
            agg = agg.reset_index()
            
        return agg, eda_slice
    
    if group_level == 'Damaged Component':
        eda_slice['ID_JOB']  = eda_slice['Shortage'] + eda_slice['Original Job Number'] + eda_slice['Reason']
        eda_slice = pd.merge(eda_slice, text_component.loc[:, text_component.columns != 'Reason'],'left',on = ['ID_JOB'])

    agg_cost_values = eda_slice.drop_duplicates([group_level,'Shortage','Shortage Cost Spread'])     
    size =  len(eda_slice.drop_duplicates([group_level,'Shortage','Original Job Number','Reason']))  
    agg_cost_values =  agg_cost_values.groupby(group_level)['Shortage Cost Spread'].sum().sort_values(ascending=False).reset_index() # group by
    # print(agg_cost_values)
    agg_cost_values['Cost Spread %'] = agg_cost_values['Shortage Cost Spread']/agg_cost_values['Shortage Cost Spread'].sum()*100
    agg_cost_values = agg_cost_values[agg_cost_values[group_level]!='missing']
    agg_cost_values[group_level] = agg_cost_values[group_level].astype(str)
    agg_cost_values['Shortage Cost Spread'] = agg_cost_values['Shortage Cost Spread'].round()
    agg_cost_values['Cost Spread %'] = agg_cost_values['Cost Spread %'].round()
    # cat_spread = cat_spread.iloc[:5]
    
    m1, m2, m3 = st.columns(3)
    
    with m1:
        st.metric(label="Level", value=level)
    with m2:
        st.metric(label="Total Shortages", value=size)
    with m3:
        cost = round(agg_cost_values['Shortage Cost Spread'].sum())
        st.metric(label="Total Cost", value= "${:,.0f}".format(cost))
    
    return agg_cost_values,eda_slice

@st.cache_data
def regression_importance(reg_slice, filter_dim, filter_val,level,columns):
    
    target = 'logy_target'
    # global columns1, columns2, columns3
    
    slice_backup = reg_slice.copy()
    
    if filter_dim=='Complexity':  
        filter_val = float(filter_val)    
    
    if level > 1:
        # print(filter_dim,filter_val)
        # print(reg_slice[filter_dim])
        reg_slice =  reg_slice[(reg_slice[filter_dim]==filter_val)] 
        
    
    columns = columns + [target]
    X = reg_slice[columns]
    X = X.dropna(subset=[target])
    X = X.drop_duplicates()
    X =pd.get_dummies(X)
    y = X[target]
    X = X.drop(target, axis=1)
    
    
    # print(X)

    # Split the dataset into training and testing sets
    print("Entry shape ",slice_backup.shape)
    print(f"Filtered on {filter_dim} - {filter_val} shape ",reg_slice.shape)
    print("Deduped on selected columns shape ",X.shape)
    
    if len(X) < 5:
        print("Exiting Out")
        print("Entry shape ",slice_backup.shape)
        print(f"Filtered on {filter_dim} - {filter_val} shape ",reg_slice.shape)
        print("Deduped on selected columns shape ",X.shape)
        return 'Few', None, None, None
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    feature_importances = model.feature_importances_
    importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
    importance_df['Column'] = importance_df['Feature'].str.split("_").apply(lambda x: '_'.join(x[:-1]) if len(x)>1 else x[0] )
    # importance_df['Value'] = importance_df['Feature'].str.split("_").apply(lambda x: x[-1] if len(x) > 1 else 'none')
    feature_importance_df = importance_df.groupby('Column')['Importance'].sum().sort_values(ascending=False).reset_index()
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    feature_importance_df['Importance'] = feature_importance_df['Importance'].apply(lambda x: "{:f}".format(x))
    feature_importance_df['Importance'] = feature_importance_df['Importance'].apply(lambda x: str(round(float(x)*100))+'%')
    top_features = feature_importance_df.copy()

    return top_features,reg_slice, filter_dim, filter_val