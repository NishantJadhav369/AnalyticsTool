# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import catboost
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import plotly.express as px  # interactive charts
import streamlit as st 
pd.set_option('display.max_columns', None)
plt.rcParams["figure.figsize"] = (20,4)

path1 = '../data/Shortage_regression.xlsx'
path2 = '../data/Shortage_Graphs.xlsx'
@st.cache_data
def load_data(path1,path2):
    df1 = pd.read_excel(path1,dtype={'PHASE':str})
    df2 = pd.read_excel(path2,dtype={'PHASE':str})
    return df1,df2

regression,claims = load_data(path1,path2)

# dashboard title
st.title("Shortage Claims Simulation")

@st.cache_data
def cost_graph(eda_slice, filter_dim, filter_val, group_level,level):
    
    global table, cost_eff_pct
    
    if filter_dim or filter_val:
        eda_slice = eda_slice[eda_slice[filter_dim] == filter_val] # EDA slice 1
        
    cat_spread = eda_slice.drop_duplicates([group_level,'SHORTAGE','SHORTAGE_COST_SPREAD'])     
    size = len(cat_spread)
    
    cat_spread =  cat_spread.groupby(group_level)['SHORTAGE_COST_SPREAD'].sum().sort_values(ascending=False).reset_index() # group by
    cat_spread['Cost Spread %'] = cat_spread['SHORTAGE_COST_SPREAD']/cat_spread['SHORTAGE_COST_SPREAD'].sum()*100
    cat_spread = cat_spread[cat_spread[group_level]!='missing']
    cat_spread[group_level] = cat_spread[group_level].astype(str)
    cat_spread['SHORTAGE_COST_SPREAD'] = cat_spread['SHORTAGE_COST_SPREAD'].round()
    cat_spread['Cost Spread %'] = cat_spread['Cost Spread %'].round()
    cat_spread = cat_spread.iloc[:5]
    
    m1, m2, m3 = st.columns(3)
    
    with m1:
        st.metric(label="Level", value=level)
    with m2:
        st.metric(label="Total Size", value=size)
    with m3:
        cost = round(cat_spread['SHORTAGE_COST_SPREAD'].sum())
        st.metric(label="Total Cost", value= "${:,.0f}".format(cost))
    
    return cat_spread,eda_slice

@st.cache_data
def regression_importance(reg_slice, filter_dim, filter_val,level):
    
    target = 'logy_target'
    columns1 = ['COMPANY','PHASE','BRAND','Major Category','COMPLEXITY','OPL','ORIGINAL_MFG_PLANT', target]
    columns2 = ['Minor Category','GL_PERIOD_MONTH','RELEASE_DATE_MONTH', 'SALES_REGION_ID','SQUAD','TEAM','REASON','BSR'] #,'CHECKER','DETAILER'
    columns3 = ['SALES_DISTRICT_ID', 'PIECEMARK','CUSTOMER_NAME']
    if level ==0:
        columns = columns1 
    elif level ==1:
        columns = columns1 + columns2
    else:
        columns = columns1 + columns2 + columns3
    

    # print(reg_slice.shape)
    reg_slice =  reg_slice[(reg_slice[filter_dim]==filter_val)] 
    # print(reg_slice.shape)
    
    X = reg_slice[columns]
    X = X.dropna(subset=[target])
    X = X.drop_duplicates()
    X =pd.get_dummies(X)
    y = X[target]
    X = X.drop(target, axis=1)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)


    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    feature_importances = model.feature_importances_
    importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
    importance_df['Column'] = importance_df['Feature'].str.split("_").apply(lambda x: '_'.join(x[:-1]) if len(x)>1 else x[0] )
    importance_df['Value'] = importance_df['Feature'].str.split("_").apply(lambda x: x[-1] if len(x) > 1 else 'none')
    feature_importance_df = importance_df.groupby('Column')['Importance'].sum().sort_values(ascending=False).reset_index()
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    feature_importance_df['Importance'] = feature_importance_df['Importance'].apply(lambda x: "{:f}".format(x))
    feature_importance_df['Importance'] = feature_importance_df['Importance'].apply(lambda x: str(round(float(x)*100))+'%')
    top_features = feature_importance_df.head(5)

    return top_features,reg_slice, filter_dim, filter_val

tab1, tab2 = st.tabs(['Major Category','Brand'])

with tab1:
    group_level = 'Major Category'
    reg_slice = regression.copy()
    eda_slice = claims.copy()
    level=0
    iter_limit = 7
    value_entry = True
    cost_eff_pct =1

    table = pd.DataFrame(columns=['Dimension','Value','Cost','% Total Cost']) 

    cost_result,eda_slice = cost_graph(eda_slice, None,None,group_level,level) # call
    fig = px.bar(cost_result,group_level,'SHORTAGE_COST_SPREAD',color='SHORTAGE_COST_SPREAD',text='SHORTAGE_COST_SPREAD',title=f'Shortage Cost Spread by {group_level}')
    fig.update_traces(texttemplate='$%{text:,.2s}')
    fig.update_layout(margin=dict(t=20))
    fig.update_layout(title=dict(x=0.3, y=1, xanchor='center', yanchor='top'))
    fig.update_yaxes(showgrid=False)
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    st.write(fig)
    
    cost_val = cost_result[group_level].iloc[0]
    cost_pct =  cost_result['Cost Spread %'].iloc[0]/100
    print(cost_pct)
    cost_amount = cost_result['SHORTAGE_COST_SPREAD'].iloc[0]    
    cost_eff_pct = cost_eff_pct*cost_pct
    record ={'Level':[level],'Dimension':[group_level],'Value': [cost_val],'Cost':[cost_amount] , '% Total Cost':[cost_eff_pct*100] }
    record = pd.DataFrame(record)
    table = pd.concat([table,record])
    table = table.reset_index(drop=True)
    # table = table.set_index('Level')
    table = table[['Dimension','Value','Cost','% Total Cost']]
    # st.subheader('Summary', divider=True)
    # st.dataframe(table)

    for i in range(iter_limit):    
        if value_entry: #avoid duplication of radio buttons
            value = st.radio( "Based on the top 5 categories contributing to the Shortage Cost Spread, **select a category** to identify the driving factors for chosen category using regression analysis.", cost_result[group_level].unique().tolist(), horizontal=True,index=None)
            value_entry = False
            # print("MY VAL -------", value)
            
            # print(group_level,value)

            # level_cost = cost_result[cost_result[group_level]==value]['SHORTAGE_COST_SPREAD'].values[0]
            # level_cost_pct = cost_result[cost_result[group_level]==value]['Cost Spread %'].values[0]
            
            # print(level_cost,level_cost_pct)

        if value and group_level:
            regression_output,reg_slice, filter_dim, filter_val = regression_importance(reg_slice,group_level,value,level) 
            fig = px.bar(regression_output,'Importance','Column',orientation='h',text='Importance',title=f'Regression analysis for {value} claims')
            fig.update_layout(title=dict( yanchor='top'))
            st.write(fig)
            
            level+=1
            st.divider()
            
            # if group_level_entry:
            group_level = st.radio(f"Based on the top 5 dimensions driving {value} claims, **select a dimension** to look at Total Cost Spread Distribution", regression_output['Column'].unique().tolist(), horizontal=True, index=None) 
            # group_level_entry = False
            # print("MY GROUP -------", group_level)
            
            

            if group_level:
                cost_result,eda_slice = cost_graph(eda_slice, filter_dim,filter_val,group_level,level) # call
                fig = px.bar(cost_result,group_level,'SHORTAGE_COST_SPREAD',color='SHORTAGE_COST_SPREAD',text='SHORTAGE_COST_SPREAD',title=f'Shortage Cost Spread by {group_level}')
                fig.update_traces(texttemplate='$%{text:,.2s}')
                fig.update_layout(margin=dict(t=20))
                fig.update_layout(title=dict(x=0.3, y=1, xanchor='center', yanchor='top'))
                st.write(fig)
                
                cost_val = cost_result[group_level].iloc[0]
                cost_pct =  cost_result['Cost Spread %'].iloc[0]/100
                print(cost_pct)
                cost_amount = cost_result['SHORTAGE_COST_SPREAD'].iloc[0]    
                cost_eff_pct = cost_eff_pct*cost_pct
                record ={'Level':[level],'Dimension':[group_level],'Value': [cost_val],'Cost':[cost_amount] , '% Total Cost':[cost_eff_pct*100] }
                record = pd.DataFrame(record)
                table = pd.concat([table,record])
                table = table.reset_index(drop=True)
                # table = table.set_index('Level')
                table = table[['Dimension','Value','Cost','% Total Cost']]
                st.subheader('Path Summary', divider=True)
                st.dataframe(table)
                
                value_entry = True
                # group_level_entry = True
                
                if len(eda_slice) <=50:
                    
                    st.markdown("### Detailed Data View")
                    st.dataframe(eda_slice.reset_index(drop=True))
                    break

# with tab2:
#     group_level = 'BRAND'
#     reg_slice = regression.copy()
#     eda_slice = claims.copy()
#     level=0
#     iter_limit = 7
#     value_entry = True
#     # group_level_entry = True

#     cost_result,eda_slice = cost_graph(eda_slice, None,None,group_level,level)
#     fig = px.bar(cost_result,group_level,'Cost Spread %',color='Cost Spread %')
#     st.write(fig)


#     for i in range(iter_limit):    
#         if value_entry: 
#             value = st.radio( "Choose Value", ['Unselected'] + cost_result[group_level].unique().tolist(),horizontal=True)
#             print(value,group_level)
#             value_entry = False

#         if value and group_level:
#             print("Regression ", i)
#             regression_output,reg_slice, filter_dim, filter_val = regression_importance(reg_slice,group_level,value,level)
#             fig = px.bar(regression_output,'Importance','Column',orientation='h')
#             st.write(fig)
#             level+=1

#             # if group_level_entry:
#             group_level = st.radio( "Choose Dimension",  ['Unselected'] + regression_output['Column'].unique().tolist()) 
#             # group_level_entry = False

#             if group_level :
#                 cost_result,eda_slice = cost_graph(eda_slice, filter_dim,filter_val,group_level,level)
#                 fig = px.bar(cost_result,group_level,'Cost Spread %',color='Cost Spread %')
#                 st.write(fig)
#                 value_entry = True
#                 # group_level_entry = True
                
#                 if len(eda_slice) <=10:
                    
#                     st.markdown("### Detailed Data View")
#                     st.dataframe(eda_slice.reset_index(drop=True))
#                     break