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
from services.utils import load_data, eda_table, regression_importance
pd.set_option('display.max_columns', None)
plt.rcParams["figure.figsize"] = (20,4)

path1 = '../data/Shortage_regression.xlsx'
path2 = '../data/Shortage_Graphs.xlsx'
path3 = '../data/ship_piecemark_region_district.xlsx'
path4 = '../data/all_shortage_notes_damaged_component_gpt4.csv'
regression,claims,ship,text_component = load_data(path1,path2,path3,path4)
# dashboard title
st.title("Shortage Claims Simulation")
group_level = 'Major Category'
level1 = ['COMPANY','Major Category','COMPLEXITY']
level2 = ['Minor Category','GL_PERIOD_MONTH','RELEASE_DATE_MONTH', 'SALES_REGION_ID','SQUAD','TEAM','REASON','BSR', 'BRAND', 'ORIGINAL_MFG_PLANT', 'SALES_DISTRICT_ID','Claim Component','CUSTOMER_NAME']
level3 = ['PIECEMARK']
sel_columns1 = st.multiselect('Choose **Level 1** columns to perform AI analytics and identify significant driving factors',level1,level1)
submit1 = st.button('Submit')
print(submit1)
reg_slice = regression.copy()
eda_slice = claims.copy()
iter_limit = 7
value_entry = True
cost_eff_pct =1
table = pd.DataFrame(columns=['Dimension','Value','Cost','% Total Cost']) 

# LEVEL 1
level =1
# group_level = 'None'
# value = 'Shortage'
options = level1
default_level = level1
if submit1: # Enter if value and group_level is selected by user
        # run regression
        regression_output,reg_slice, filter_dim, filter_val = regression_importance(reg_slice,"None","Shortage",level,sel_columns1) # reg call        
        fig = px.bar(regression_output,'Importance','Column',orientation='h',text='Importance',title=f'Regression analysis for {"Shortage"} claims')
        fig.update_layout(title=dict( yanchor='top'))
        st.write(fig)
        group_level = st.radio(f"Based on the top dimensions driving {'Shortage'} claims, **select a dimension** to look at Total Cost Spread Distribution", regression_output['Column'].unique().tolist(), horizontal=True, index=None) 
        if group_level:
            # get aggregated table
            cost_result,eda_slice = eda_table(eda_slice,text_component,ship, filter_dim,filter_val,group_level,level) # eda call
            top_results = cost_result[:10]
            # print("RETURNED ",group_level)
            # print(top_results)
            fig = px.bar(top_results,group_level,'SHORTAGE_COST_SPREAD',color='SHORTAGE_COST_SPREAD',text='SHORTAGE_COST_SPREAD',title=f'Shortage Cost Spread by {group_level}')
            fig.update_traces(texttemplate='$%{text:,.2s}')
            fig.update_layout(margin=dict(t=20))
            fig.update_layout(title=dict(x=0.3, y=1, xanchor='center', yanchor='top'))
            st.write(fig)
            st.divider()
            value = st.radio( "Based on the top 5 values contributing to the Shortage Cost Spread, **select a value** to identify the driving factors for chosen category using regression analysis.", top_results[group_level].unique().tolist(), horizontal=True,index=None)
            if value:
                # Path Summary table
                sel_record = top_results[top_results[group_level]==value]
                cost_val = sel_record[group_level].iloc[0]
                cost_pct =  sel_record['Cost Spread %'].iloc[0]/100
                cost_amount = sel_record['SHORTAGE_COST_SPREAD'].iloc[0]    
                cost_eff_pct = cost_eff_pct*cost_pct
                record ={'Level':[level],'Dimension':[group_level],'Value': [cost_val],'Cost':[cost_amount] , '% Total Cost':[cost_eff_pct*100] }
                record = pd.DataFrame(record)
                table = pd.concat([table,record])
                table = table.reset_index(drop=True)
                # table = table.set_index('Level')
                table = table[['Dimension','Value','Cost','% Total Cost']]
                st.subheader('Path Summary', divider=True)
                st.dataframe(table)
                
if submit1 and group_level and value:              
    # LEVEL 2
    default_level = ['Minor Category','ORIGINAL_MFG_PLANT','BSR','REASON','Claim Component']
    level = 2
    options = level2
    sel_columns2 = st.multiselect(f'Choose **Level {level}** columns to perform AI analytics and identify significant driving factors',options,default_level,key=f'multiselect_{1}')
    submit2 = st.button('Submit')
    print('Submit2 ', submit2)

    if submit2:

        for i in range(1, 4):  
            print(i,select_flag)
            level = 2
            options = level2
            # sel_columns2 = st.multiselect(f'Choose **Level {level}** columns to perform AI analytics and identify significant driving factors',options,default_level,key=f'multiselect_{i}')

            if st.button('Submit') and value and group_level: # Enter if value and group_level is selected by user
                # run regression
                print("RUN REGRESSION")
                regression_output,reg_slice, filter_dim, filter_val = regression_importance(reg_slice,group_level,value,level,sel_columns2) # reg call        
                fig = px.bar(regression_output,'Importance','Column',orientation='h',text='Importance',title=f'Regression analysis for {value} claims')
                fig.update_layout(title=dict( yanchor='top'))
                st.write(fig)
                group_level = st.radio(f"Based on the top dimensions driving {value} claims, **select a dimension** to look at Total Cost Spread Distribution", regression_output['Column'].unique().tolist(), horizontal=True, index=None) 
                if group_level:
                    # get aggregated table
                    cost_result,eda_slice = eda_table(eda_slice,text_component,ship, filter_dim,filter_val,group_level,level) # eda call
                    top_results = cost_result[:10]
                    fig = px.bar(top_results,group_level,'SHORTAGE_COST_SPREAD',color='SHORTAGE_COST_SPREAD',text='SHORTAGE_COST_SPREAD',title=f'Shortage Cost Spread by {group_level}')
                    fig.update_traces(texttemplate='$%{text:,.2s}')
                    fig.update_layout(margin=dict(t=20))
                    fig.update_layout(title=dict(x=0.3, y=1, xanchor='center', yanchor='top'))
                    st.write(fig)
                    st.divider()
                    value = st.radio( "Based on the top 5 values contributing to the Shortage Cost Spread, **select a value** to identify the driving factors for chosen category using regression analysis.", top_results[group_level].unique().tolist(), horizontal=True,index=None)
                    if value:
                        # Path Summary table
                        sel_record = top_results[top_results[group_level]==value]
                        cost_val = sel_record[group_level].iloc[0]
                        cost_pct =  sel_record['Cost Spread %'].iloc[0]/100
                        cost_amount = sel_record['SHORTAGE_COST_SPREAD'].iloc[0]    
                        cost_eff_pct = cost_eff_pct*cost_pct
                        record ={'Level':[level],'Dimension':[group_level],'Value': [cost_val],'Cost':[cost_amount] , '% Total Cost':[cost_eff_pct*100] }
                        record = pd.DataFrame(record)
                        table = pd.concat([table,record])
                        table = table.reset_index(drop=True)
                        table = table[['Dimension','Value','Cost','% Total Cost']]
                        st.subheader('Path Summary', divider=True)
                        st.dataframe(table)
                        default_level.remove(group_level)

            
# LEVEL 3
#     if group_level == 'PIECEMARK':
#         top_results['PIECEMARK'] = top_results['PIECEMARK'].astype(str)
#         fig2 = px.bar(top_results,'PIECEMARK','QUANTITY')
#         # print("WHY")
#         st.write(fig2) 

#     if (len(eda_slice) <=10) or ( level == 3):
#         st.markdown("### Detailed Data View")
#         st.dataframe(eda_slice.reset_index(drop=True))
#         eda_slice.to_excel('../data/example.xlsx')

