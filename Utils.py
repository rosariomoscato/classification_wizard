import pandas as pd
import streamlit as st
import os
import base64
from datetime import datetime


def download(df, filename):  # Downloading DataFrame
  csv = df.to_csv(index=False)
  b64 = base64.b64encode(csv.encode()).decode()
  href = (
    f'<a href="data:file/csv;base64,{b64}" download="%s.csv">Download csv file</a>'
    % (filename))
  return href


def file_upload(name):
  uploaded_file = st.sidebar.file_uploader('%s' % (name),
                                           key='%s' % (name),
                                           accept_multiple_files=False)
  content = False
  if uploaded_file is not None:
    try:
      uploaded_df = pd.read_csv(uploaded_file)
      content = True
      return content, uploaded_df
    except:
      try:
        uploaded_df = pd.read_excel(uploaded_file)
        content = True
        return content, uploaded_df
      except:
        st.error(
          'Please ensure file is .csv or .xlsx format and/or reupload file')
        return content, None
  else:
    return content, None
