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
from datetime import datetime as dt
import datetime
pd.set_option('display.max_columns', None)
plt.rcParams["figure.figsize"] = (20,4)

path1 = '../data/Shortage_regression.xlsx'
path2 = '../data/Shortage_Graphs.xlsx'
path3 = '../data/ship_piecemark_region_district.xlsx'
path4 = '../data/all_shortage_notes_damaged_component_gpt4.csv'

regression,claims,ship,text_component = load_data(path1,path2,path3,path4)
# regression.rename(columns={'COMPANY': 'COMPLEXITY', 'GL_PERIOD_MONTH'
renamer = {'COMPANY': 'Company','COMPLEXITY':'Complexity', 'GL_PERIOD': 'GL Period','RELEASE_DATE':'Release Date','GL_PERIOD_MONTH': 'GL Period Month', 'RELEASE_DATE_MONTH' :'Release Date Month', 'SALES_REGION_ID':'Sales Region ID', 'SQUAD':'Squad', 'TEAM':'Team', 'REASON':'Reason', 'BSR':'BSR', 'BRAND':'Brand', 'ORIGINAL_MFG_PLANT':'Original Mfg Plant', 'SALES_DISTRICT_ID':'Sales Region ID', 'CUSTOMER_NAME':'Customer Name','PIECEMARK':'Piecemark','Claim Component':'Damaged Component','SHORTAGE_COST_SPREAD':'Shortage Cost Spread','SHORTAGE':'Shortage','ORIGINAL_JOB_NUM':'Original Job Number','QUANTITY':'Quantity'}

claims.rename(columns = renamer, inplace=True)
regression.rename(columns = renamer, inplace=True)
ship.rename(columns = renamer, inplace=True)
text_component.rename(columns = renamer, inplace=True)


# dashboard title
st.title("Shortage Claims Simulation")

# filter1, filter2, filter3, kpi1 = st.columns(4)

# with filter1:
#     vertical = st.selectbox("Data Filter", ['GL Period','Release Date']) 
#     max_date = claims[claims['Release Date']!='missing']['Release Date'].max() 
#     min_date = claims[claims['Release Date']!='missing']['Release Date'].min()

# with filter2:
#     start_date = st.date_input("Start Date", min_value=min_date, max_value=max_date,value=min_date,format="MM/DD/YYYY")
#     start_date = str(start_date).split()[0]
#     start_date = dt.strptime(start_date,"%Y-%m-%d") 

# with filter3:
#     end_date = st.date_input("End Date", min_value=min_date, max_value=max_date,value=max_date,format="MM/DD/YYYY")
#     end_date = str(end_date).split()[0]
#     end_date =  dt.strptime(end_date,"%Y-%m-%d") 

#     interval = end_date - start_date
    
#     # fill in those three columns with respective metrics or KPIs
#     kpi1.metric(
#         label="Interval Period (Days)",
#         value=interval.days
#     )
    
# print(type(claims[vertical].values[0]))
    
    
# print('Min date is here ',min_date)    
# print(type(min_date))
# date_range = st.slider( "Start Date", value= min_date, format="MM/DD/YY - hh:mm")#,format="MM/DD/YYYY")  



    
# print('Start Time',start_time)
# print(claims[vertical].values[0]) 

# start_date = st.slider("Select start date",
#                       int((datetime.datetime.today() - datetime.timedelta(days=30)).timestamp()),
#                       format="YYYY-MM-DD")

# print(' ------------ ' , start_date)
    
# print(curr_start,curr_end)
# claims = claims[(claims[vertical] >= start_date) & (claims[vertical] <= end_date)]      
    

# tab1, tab2 = st.tabs(['Major Category','Brand'])

# with tab1:
group_level = 'Major Category'

level1 = ['Company','Major Category','Complexity']
level2 = ['Minor Category','GL Period Month','Release Date Month', 'Sales Region ID','Squad','Team','Reason','BSR', 'Brand', 'Original Mfg Plant', 'Sales District ID','Damaged Component','Customer Name']


sel_columns = st.multiselect('Choose **Level 1** columns to perform AI analytics and identify significant driving factors',level1,level1)

reg_slice = regression.copy() # [(regression[vertical] >= start_date) & (regression[vertical] <= end_date)].copy()
eda_slice = claims.copy() #[(claims[vertical] >= start_date) & (claims[vertical] <= end_date)].copy()
iter_limit = 7
value_entry = True
cost_eff_pct =1
time_agg_type =  None


table = pd.DataFrame(columns=['Dimension','Value','Cost','% Total Cost']) 

curr_max_level= 0

for level in range(1, 11):    
    
    if level==1:
        group_level = 'None'
        value = 'Shortage'
        options = level1
        default_level = level1
        
    print(f"----{level} ------{group_level} -- {value}")
        
    if value and group_level: # Enter if value and group_level is selected by user
        # run regression
        regression_output,reg_slice, filter_dim, filter_val = regression_importance(reg_slice,group_level,value,level,sel_columns) # reg call
        print(filter_dim,filter_val)
        if level > 2 and (reg_slice is not None):
            reg_slice.to_excel(f'../data/eg3_detailed_{level}_{group_level}.xlsx')
                    
        if type(regression_output)==str:
            print("Very few claims available for Regression using the mentioned columns")
            st.markdown("Based on the above chosen columns there are few claims available in data slice for further analysis. Please use another set of features.")
            # st.markdown("### Detailed Data View")
            # detailed_view = eda_slice.loc[:, eda_slice.columns != 'UID'][sel_columns + ['SHORTAGE_COST_SPREAD']]
            # st.dataframe(detailed_view.reset_index(drop=True))
            break
        else:
            regression_output.to_excel(f'../data/eg2_importance_{level}_{group_level}.xlsx')
            
        fig = px.bar(regression_output,'Importance','Column',orientation='h',text='Importance',title=f'Regression analysis for {value} claims')
        fig.update_layout(title=dict( yanchor='top'))
        st.write(fig)

        group_level = st.radio(f"Based on the top dimensions driving {value} claims, **select a dimension** to look at Total Cost Spread Distribution", regression_output['Column'].unique().tolist(), horizontal=True, index=None) 

        if group_level:
            # get aggregated table
            if (group_level == 'GL Period Month') or (group_level == 'Release Date Month'):
                if group_level == 'GL Period Month':
                    group_level = 'GL Period'
                elif group_level == 'Release Date Month':
                    group_level = 'Release Date'
                    
                if group_level == 'GL Period':
                    # group_level = 'GL Period'
                    time_agg_type = st.radio( "Select Periodicity",['MoM','QoQ','YoY'],index=0, horizontal=True)
                elif group_level == 'Release Date':
                    # group_level = 'Release Date'
                    time_agg_type = st.radio( "Select Periodicity",['WoW','MoM','QoQ','YoY'],index=1, horizontal=True)                
                
            cost_result,eda_slice = eda_table(eda_slice,text_component,ship, filter_dim,filter_val,group_level,level,time_agg_type) # eda call
            # cost_result.to_excel(f'../data/eg2_graph_{level}_{group_level}.xlsx')
            
            if (group_level == 'GL Period') or (group_level == 'Release Date'):
                top_results = cost_result
                value_options = top_results[group_level].unique().tolist()
            else:
                top_results = cost_result[:10]
                value_options = top_results[group_level].unique().tolist()

            
            fig = px.bar(top_results,group_level,'Shortage Cost Spread',color='Shortage Cost Spread',text='Shortage Cost Spread',title=f'Shortage Cost Spread by {group_level}')
            fig.update_traces(texttemplate='$%{text:,.2s}')
            fig.update_layout(margin=dict(t=20))
            fig.update_layout(title=dict(x=0.3, y=1, xanchor='center', yanchor='top'))
            st.write(fig)
            
            
            st.divider()
            
            if group_level not in [ 'GL Period','Release Date']:
                value = st.radio( "Based on the top values contributing to the Shortage Cost Spread, **select a Value** to identify the driving factors for chosen category using regression analysis.", value_options, horizontal=True,index=None)
            else: 
                st.markdown("**Select Date Range** to identify the driving factors for chosen period using regression analysis.")
                filter1, filter2, kpi1 = st.columns(3)
                max_date = eda_slice[group_level].max() 
                min_date = eda_slice[group_level].min()
                with filter1:
                    start_date = st.date_input("Start Date", min_value=min_date, max_value=max_date,value=min_date,format="MM/DD/YYYY")
                    start_date = str(start_date).split()[0]
                    start_date = dt.strptime(start_date,"%Y-%m-%d") 

                with filter2:
                    end_date = st.date_input("End Date", min_value=min_date, max_value=max_date,value=max_date,format="MM/DD/YYYY")
                    end_date = str(end_date).split()[0]
                    end_date =  dt.strptime(end_date,"%Y-%m-%d") 

                    interval = end_date - start_date

                    # fill in those three columns with respective metrics or KPIs
                    kpi1.metric(
                        label="Interval Period (Days)",
                        value=interval.days
                    ) 
                    
                eda_slice = eda_slice[(eda_slice[group_level] >= start_date) & (eda_slice[group_level] <= end_date)]       

                # value = st.radio( "Based on the top values contributing to the Shortage Cost Spread, **select a value** to identify the driving factors for chosen category using regression analysis.", value_options, horizontal=True,index=None)
                
                
            if value and (group_level not in [ 'GL Period','Release Date']):
                # Path Summary table
                sel_record = top_results[top_results[group_level]==value]
                cost_val = sel_record[group_level].iloc[0]
                cost_pct =  sel_record['Cost Spread %'].iloc[0]/100
                cost_amount = sel_record['Shortage Cost Spread'].iloc[0]    
                cost_eff_pct = cost_eff_pct*cost_pct
                record ={'Level':[level],'Dimension':[group_level],'Value': [cost_val],'Cost':[cost_amount] , '% Total Cost':[cost_eff_pct*100] }
                record = pd.DataFrame(record)
                table = pd.concat([table,record])
                table = table.reset_index(drop=True)
                # table = table.set_index('Level')
                table = table[['Dimension','Value','Cost','% Total Cost']]
                st.subheader('Path Summary', divider=True)
                st.dataframe(table)
            else: # WIP
                cost_val = str(start_date).split()[0] + " to " + str(end_date).split()[0]
                cost_amount = 0
                cost_eff_pct =  0
                record ={'Level':[level],'Dimension':[group_level],'Value': [cost_val],'Cost':[cost_amount] , '% Total Cost':[cost_eff_pct*100] }
                record = pd.DataFrame(record)
                table = pd.concat([table,record])
                table = table.reset_index(drop=True)
                # table = table.set_index('Level')
                table = table[['Dimension','Value','Cost','% Total Cost']]
                st.subheader('Path Summary', divider=True)
                st.dataframe(table)
            
            
            if value and level>1 and (group_level =='Damaged Component'):
                temp_slice = eda_slice[eda_slice[group_level]==value].copy()
                cost_result,eda_slice = eda_table(temp_slice,text_component,ship, filter_dim,filter_val,'Piecemark',level, time_agg_type) # eda call
                top_results = cost_result[:10]
                top_results['Piecemark'] = top_results['Piecemark'].astype(str)
                fig2 = px.bar(top_results,'Piecemark','Quantity',text='Quantity',color='Quantity',title=f'Quantity by Piecemark for {value}')
                fig2.update_layout(margin=dict(t=20))
                fig2.update_layout(title=dict(x=0.5, y=1, xanchor='center', yanchor='top'))
                st.write(fig2) 
                # cost_result.to_excel(f'../data/eg2_peicemark_{level}_{group_level}.xlsx')
                

            options = [ option for option in level2 if option not in table['Dimension'].values ]
            if level ==1:
                default_level = ['Minor Category','Original Mfg Plant','BSR','Reason','Damaged Component','Release Date Month','GL Period Month','Customer Name']
            else:
                default_level = [ col for col in sel_columns if col not in table['Dimension'].values ]
            
            sel_columns = st.multiselect(f'Choose **Level {level + 1}** columns to perform AI analytics and identify significant driving factors',options,default_level)
            # print(sel_columns)
            
            
            

            
