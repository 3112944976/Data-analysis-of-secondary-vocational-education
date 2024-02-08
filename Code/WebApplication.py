#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vector
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
import streamlit as st
import pandas as pd

def Get_Session():
    return SparkSession.builder.appName('WebApplication').getOrCreate()
def LR_Go(saprk,inputcols,labelcol):
    df = saprk.read.csv("hdfs://master:9000/home/hadoop/data/edu/GeneralData.csv", inferSchema=True, header = True)
    vec = VectorAssembler(inputCols=inputcols, outputCol='features')
    features_df = vec.transform(df)
    model_df = features_df.select('features',labelcol)
    train_df, test_df = model_df.randomSplit([0.7, 0.3],seed=123)
    lin_reg = LinearRegression(featuresCol='features',labelCol=labelcol)
    lr_model = lin_reg.fit(train_df)
    predictions = lr_model.transform(test_df)
    GO = predictions.toPandas()
    GO.to_csv('Predictions.csv')
      

st.set_page_config(page_title='中职院校招生数预测')
st.header('中职院校招生数预测')
st.subheader('Forecast of enrollment in Secondary Vocational Colleges')

st.markdown('# 1、影响因素选择')
opts = ('招生数','在校学生数','毕业生数','获得职业资格证书毕业生数','预计毕业生数','教职工总数','专任教师数','普通中等专业学校数','成人中等专业学校数')
inputcols =  st.multiselect('',opts)

st.markdown('# 2、预测变量选择')
labelcols = st.selectbox('', options=opts)

st.markdown('# 3、预测结果')
sess = Get_Session()
st.write("Spark集群准备就绪，节点构建完成！")
result = st.button("点击运行程序")
if result:
    LR_Go(sess,list(inputcols),str(labelcols))
    GOO = pd.read_csv('Predictions.csv')
    st.dataframe(GOO)

