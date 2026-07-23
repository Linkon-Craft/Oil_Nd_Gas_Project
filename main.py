import streamlit as st


st.set_page_config(
    page_title="Oil & Gas Classification",
    page_icon="🛢️",
    layout="wide"
)

pg = st.navigation([
    st.Page("oil_nd_gas.py", title="Oil & Gas Classification", icon="🛢️"),
    st.Page("app.py", title="Test Oil and Gas CSV Dataset", icon="🌊"),
    st.Page("single_predict.py", title="Single Predict Oil Field Location", icon="🤖")
])

pg.run()