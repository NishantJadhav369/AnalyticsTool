import streamlit as st

st.title("CBB Analytics Tool")

st.header('INTRODUCTION', divider=True)

st.markdown('''The simulation follows a top-down approach, starting with higher-level columns and progressively delving into greater granularity during the analysis. The column categorization is organized into two distinct levels based on granularity. 

Level 1 comprises columns offering a broader perspective, while Level 2 ... Level N include columns providing detailed information at a granular level.

**Level 1** - *Company, Major Category, Complexity*

**Level 2** - *Minor Category, GL Period Month, Release Date Month, Sales Region ID, Sales District ID, Squad, Team, Reason, BSR, Brand, Original Mfg Plant, Customer Name, Claim Component*

**Level N** - *Minor Category, GL Period Month, Release Date Month, Sales Region ID, Sales District ID, Squad, Team, Reason, BSR, Brand, Original Mfg Plant, Customer Name, Claim Component*

Additionally, for every Claim Component user can look into Quantity by Piecemark distribution
''')

st.subheader('User Interaction Walkthrough', divider=True)

st.markdown('''
**Step 1** - Choose feature options for Level 1 Analysis to identify driving significant features using Machine Learning techniques.

**Step 2** - Based on Significance scores, select one feature to look at cost distribution across unique values. 

**Step 3** - Choose a value for identifying key features that drive claims associated with that unique value.

**Step 4** - Choose feature options for Level 2 Analysis and continue analysis to investigate distinct features and associated values.

''')
