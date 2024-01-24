import streamlit as st

st.title("CBB Analytics Tool")

st.header('INTRODUCTION', divider=True)

st.markdown('''The simulation follows a top-down approach, starting with higher-level columns and progressively delving into greater granularity during the analysis. The column categorization is organized into three levels based on granularity. The top level comprises columns offering a broader perspective, while the bottom level includes columns providing detailed information at a granular level.

**Level 1** - *Company, Phase, Brand, Major Category, Complexity, OPL, Original Mfg Plant*

**Level 2** - *Minor Category, GL Period Month, Release Date Month, Sales Region ID, Squad, Team, Reason, BSR*

**Level 3** - *Sales District ID, Piecemark, Customer Name*
''')

