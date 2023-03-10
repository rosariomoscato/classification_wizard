import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import base64

from datetime import datetime
import os
from Utils import *


def confusion_matrix_plot(data, labels):
  z = data.tolist()[::-1]
  x = labels
  y = labels[::-1]
  z_text = z

  fig = ff.create_annotated_heatmap(z,
                                    x,
                                    y,
                                    annotation_text=z_text,
                                    text=z,
                                    hoverinfo='text',
                                    colorscale='Blackbody')
  fig.update_layout(font_family="IBM Plex Sans")

  st.write(fig)


def roc_plot(data):
  fig = px.line(data, x="False Positive",
                y="True Positive")  #, title='ROC Curve')
  fig.update_layout(font_family="IBM Plex Sans")

  st.write(fig)


def svm_main():

  st.sidebar.subheader('Training Dataset')
  status, df = file_upload('Please upload a training dataset')

  if status == True:

    col_names = list(df)

    st.title('Training')
    st.subheader('Parameters')
    col1, col2, col3 = st.columns((3, 3, 2))

    with col1:
      feature_cols = st.multiselect('Please select features', col_names)
    with col2:
      label_col = st.selectbox('Please select label', col_names)
    with col3:
      test_size = st.number_input('Please enter test size', 0.01, 0.99, 0.25,
                                  0.05)

    with st.expander('Advanced Parameters'):
      col2_1, col2_2 = st.columns(2)
      with col2_1:
        C = st.number_input('Regularization parameter', 0.0, 99.0, 1.0, 1.0)
        degree = st.number_input('Degree', 0, 100, 3, 1)
        coef0 = st.number_input('Coef0', 0.0, 99.0, 0.0, 1.0)
        probability = st.radio('Probability', [True, False])
        class_weight = st.radio('Class weight', [None, 'balanced'])
        max_iter = st.number_input('Maximum iterations', -1, 100, -1, 1)
        break_ties = st.radio('Break ties', [False, True])
      with col2_2:
        kernel = st.selectbox(
          'Kernel', ['rbf', 'linear', 'poly', 'sigmoid', 'precomputed'])
        gamma = st.selectbox('Gamma', ['scale', 'auto'])
        shrinking = st.radio('Shrinking', [True, False])
        tol = st.number_input('Tolerance (1e-3)', value=1) / 1000
        verbose = st.radio('Verbose', [False, True])
        decision_function_shape = st.selectbox('Decision function shape',
                                               ['ovr', 'ovo'])
        random_state = st.radio('Random state', [None, 'Custom'])
        if random_state == 'Custom':
          random_state = st.number_input('Custom random state', 0, 99, 1, 1)

      st.markdown(
        'For further information please refer to ths [link](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)'
      )

    try:
      X = df[feature_cols]
      y = df[label_col]
      X_train, X_test, y_train, y_test = train_test_split(X,
                                                          y,
                                                          test_size=test_size,
                                                          random_state=0)
      clf = svm.SVC(C=C,
                    kernel=kernel,
                    degree=degree,
                    gamma=gamma,
                    coef0=coef0,
                    shrinking=shrinking,
                    probability=probability,
                    tol=tol,
                    class_weight=class_weight,
                    verbose=verbose,
                    max_iter=max_iter,
                    decision_function_shape=decision_function_shape,
                    break_ties=break_ties,
                    random_state=random_state)
      clf.fit(X_train, y_train)
      y_pred = clf.predict(X_test)
      cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

      st.subheader('Confusion Matrix')
      confusion_matrix_plot(cnf_matrix, list(df[label_col].unique()))

      accuracy = metrics.accuracy_score(y_test, y_pred)
      precision = metrics.precision_score(y_test, y_pred)
      recall = metrics.recall_score(y_test, y_pred)
      f1 = metrics.f1_score(y_test, y_pred)

      st.subheader('Metrics')
      col2_1, col2_2, col2_3, col2_4 = st.columns(4)

      with col2_1:
        st.info('Accuracy: **%s**' % (round(accuracy, 3)))
      with col2_2:
        st.info('Precision: **%s**' % (round(precision, 3)))
      with col2_3:
        st.info('Recall: **%s**' % (round(recall, 3)))
      with col2_4:
        st.info('F1 Score: **%s**' % (round(f1, 3)))

      try:
        y_pred_proba = clf.predict_proba(X_test)[::, 1]
        fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
        roc_data = pd.DataFrame([])
        roc_data['True Positive'] = tpr
        roc_data['False Positive'] = fpr
        st.subheader('ROC Curve')
        roc_plot(roc_data)
        auc = metrics.roc_auc_score(y_test, y_pred_proba)
        st.info('Area Under Curve: **%s**' % (round(auc, 3)))
      except:
        pass

      #Download hyperparameters
      hyperparameters = {
        'C': [C],
        'kernel': [kernel],
        'degree': [degree],
        'gamma': [gamma],
        'coef0': [coef0],
        'shrinking': [shrinking],
        'probability': [probability],
        'tol': [tol],
        'class_weight': [class_weight],
        'verbose': [verbose],
        'max_iter': [max_iter],
        'decision_function_shape': [decision_function_shape],
        'break_ties': [break_ties],
        'random_state': [random_state]
      }

      st.subheader('Download hyperparameters')
      st.markdown(download(
        pd.DataFrame(hyperparameters),
        'Classification Wizard - Support Vector Machine Classifier - Hyperparameters'
      ),
                  unsafe_allow_html=True)

      st.sidebar.subheader('Test Dataset')
      status_test, df_test = file_upload('Please upload a test dataset')

      if status_test == True:
        try:
          st.title('Testing')

          X_test_test = df_test[feature_cols]
          y_pred_test = clf.predict(X_test_test)

          X_pred = df_test.copy()
          X_pred[label_col] = y_pred_test
          X_pred = X_pred.sort_index()

          st.subheader('Predicted Labels')
          st.write(X_pred)
          st.markdown(download(
            X_pred,
            'Classification Wizard - Support Vector Machine Classifier - Predicted Labels'
          ),
                      unsafe_allow_html=True)
        except:
          st.warning(
            'Please upload a test dataset with the same feature set as the training dataset'
          )

      elif status_test == False:
        st.sidebar.warning('Please upload a test dataset')

    except:
      st.warning(
        'Please select at least one feature, a suitable binary label and appropriate advanced parameters'
      )

  # Caso in cui non viene caricato un file di training
  elif status == False:
    st.title('Welcome ????')
    st.subheader('Please use the left pane to upload your dataset')
    st.sidebar.warning('Please upload a training dataset')

  # Per scaricare un sample dataset
  st.sidebar.subheader('Sample Dataset')
  if st.sidebar.button('Download sample dataset'):
    url = 'https://raw.githubusercontent.com/rosariomoscato/classification_wizard/main/sample_datasets/diabetes_dataset_train.csv'
    csv = pd.read_csv(url)
    st.sidebar.markdown(download(csv, 'sample_dataset'),
                        unsafe_allow_html=True)

  # Donazioni
  st.sidebar.markdown(' ')
  st.sidebar.markdown('[Donate here](https://paypal.me/rosmoscato)')

  #Footer Sidebar
  st.sidebar.markdown(
    f'<div class="markdown-text-container stText" style="width: 698px;"><footer><p></p></footer><div style="font-size: 12px;">Classification Wizard v0.8</div><div style="font-size: 12px;">rosariomoscato.github.io</div></div>',
    unsafe_allow_html=True)
