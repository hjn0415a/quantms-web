import streamlit as st
from src.common.common import page_setup
from src.WorkflowTest import WorkflowTest

params = page_setup()

wf = WorkflowTest()
st.write(st.session_state["workspace"])
st.write(type(st.session_state["workspace"]))
wf.show_execution_section()
