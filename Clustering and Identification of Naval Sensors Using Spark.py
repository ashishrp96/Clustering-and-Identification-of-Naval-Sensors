import sys
import pandas as pd
from random import random
import numpy as np
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from sklearn.metrics import classification_report, confusion_matrix
from pyspark.mllib.clustering import KMeans, KMeansModel
from pyspark.ml.clustering import KMeans
import pyspark.sql.functions as f
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql.functions import monotonically_increasing_id, row_number
from pyspark.sql.types import StructType
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

#loading Data
sc=SparkContext(appName="analytics")
sqlcontext= SQLContext(sc)
spark=SparkSession.builder.appName("analytics").getOrCreate()

mydata=spark.read.format("csv").option("inferschema","true").option("header","false").load("hdfs:///hw3/train_labeled.csv",seperator=",")
#filtering data for null values     
mydata=mydata.selectExpr("_c0 as Pressure", "_c1 as Temp","_c2 as label")
mydata=mydata.dropna(subset=("Pressure","Temp"),how="all")
mydata=mydata.filter(mydata["Pressure"]>0)
#Pressure mean and stdev values
mydata.agg({'Pressure': 'mean'}).show()
mydata.agg({'Pressure': 'stddev'}).show()
mydata.agg({'Pressure': 'max'}).show()
mydata.agg({'Pressure': 'min'}).show()

#Tempearture mean and stdev values
mydata.agg({'Temp': 'mean'}).show()
mydata.agg({'Temp': 'stddev'}).show()
mydata.agg({'Temp': 'max'}).show()
mydata.agg({'Temp': 'min'}).show()

print(mydata.show(10))

#Dividing the data into categories of sensors for outliers
print("categories")

botptdf=mydata.filter(mydata["label"]==0)


profilerdf=mydata.filter(mydata["label"]==1)


gliderdf=mydata.filter(mydata["label"]==2)

#calculating bounds for Outliers
def cal_bounds(df):
    bounds = {
        c: dict(
            zip(["q1", "q3"], df.approxQuantile(c, [0.25, 0.75], 0))
        )
        for c,d in zip(df.columns, df.dtypes) if d[1] == "double"
    }
    for c in bounds:
        iqr = bounds[c]['q3'] - bounds[c]['q1']
        bounds[c]['lower'] = bounds[c]['q1'] - (iqr * 1.5)
        bounds[c]['upper'] = bounds[c]['q3'] + (iqr * 1.5)
        print(bounds)

#flagging outliers functions

def flag_outliers(df):
    bounds = {
        c: dict(
            zip(["q1", "q3"], df.approxQuantile(c, [0.25, 0.75], 0))
        )
        for c,d in zip(df.columns, df.dtypes) if d[1] == "double"
    }
    
    for c in bounds:
        iqr = bounds[c]['q3'] - bounds[c]['q1']
        bounds[c]['lower'] = bounds[c]['q1'] - (iqr * 1.5)
        bounds[c]['upper'] = bounds[c]['q3'] + (iqr * 1.5)
    
    return df.select(
    "*",
    *[
        f.when(
            f.col(c).between(bounds[c]['lower'], bounds[c]['upper']),
            0
        ).otherwise(1).alias(c+"_out") 
        for c,d in zip(df.columns, df.dtypes) if d[1] == "double"
    ]
    )

#outlier detection for each coloumns and each sensor type or label

botpt_outlier_df= flag_outliers(botptdf)
profiler_outlier_df= flag_outliers(profilerdf)
glider_outlier_df= flag_outliers(gliderdf)

botpt_outlier_df.show()
cal_bounds(botptdf)
botpt_outlier_df.groupBy('Pressure_out').count().show()
botpt_outlier_df.groupBy('Temp_out').count().show()

profiler_outlier_df.show()
cal_bounds(profilerdf)
profiler_outlier_df.groupBy('Pressure_out').count().show()
profiler_outlier_df.groupBy('Temp_out').count().show()

glider_outlier_df.show()
cal_bounds(gliderdf)
glider_outlier_df.groupBy('Pressure_out').count().show()
glider_outlier_df.groupBy('Temp_out').count().show()

mydata.groupBy("label").count().show()

#creating features for model building and building training model

vecAssembler=VectorAssembler(inputCols=["Pressure", "Temp"], outputCol="features")
kmeans_df = vecAssembler.transform(mydata).select("Pressure","Temp","label","features")
kmeans_df.show()
kmeans_df.groupBy('label').count().show()

kmeans = KMeans().setK(3).setSeed(1)
model = kmeans.fit(kmeans_df.select("features"))

#creating predictions from the model 

transformed = model.transform(kmeans_df.select("features"))
transformed.show()
transformed.groupBy('prediction').count().show()

#centers

centers = model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)

#evaluation metrics for the classification of training data

evaluator = ClusteringEvaluator()
silhouette = evaluator.evaluate(transformed)
print("Silhouette with squared euclidean distance = " + str(silhouette))

ldf=kmeans_df.select("label")
pdf=transformed.select("prediction")

ldf=ldf.withColumn("id1",monotonically_increasing_id())
pdf=pdf.withColumn("id2",monotonically_increasing_id())

evaldf=ldf.join(pdf,ldf["id1"]==pdf["id2"],"inner").drop("id1","id2")
evaldf.show()
metrics = MulticlassMetrics(evaldf.rdd.map(lambda x: tuple(map(float, x))))

cm = metrics.confusionMatrix().toArray()
labels = [int(l) for l in metrics.call('labels')]
confusion_matrix = pd.DataFrame(cm , index=labels, columns=labels)
print(confusion_matrix)

precision = metrics.precision()
recall = metrics.recall()
f1Score = metrics.fMeasure()
print("Summary Stats")
print("Precision = %s" % precision)
print("Recall = %s" % recall)
print("F1 Score = %s" % f1Score)

for label in sorted(labels):
    print("Class %s precision = %s" % (label, metrics.precision(label)))
    print("Class %s recall = %s" % (label, metrics.recall(label)))
    print("Class %s F1 Measure = %s" % (label, metrics.fMeasure(float(label),beta=1.0)))

print("Weighted recall = %s" % metrics.weightedRecall)
print("Weighted precision = %s" % metrics.weightedPrecision)
print("Weighted F(1) Score = %s" % metrics.weightedFMeasure())
print("Weighted F(0.5) Score = %s" % metrics.weightedFMeasure(beta=0.5))
print("Weighted false positive rate = %s" % metrics.weightedFalsePositiveRate)


# loading the unlabeled test data.

testdata=spark.read.format("csv").option("inferschema","true").option("header","false").load("hdfs:///hw3/test_unlabeled.csv",seperator=",")   
testdata=testdata.selectExpr("_c0 as Pressure", "_c1 as Temp")
testdata=testdata.dropna(subset=("Pressure","Temp"),how="all")
testdata=testdata.filter(testdata["Pressure"]>0)
print(testdata.count())

#Pressure mean and stdev values
testdata.agg({'Pressure': 'mean'}).show()
testdata.agg({'Pressure': 'stddev'}).show()
testdata.agg({'Pressure': 'max'}).show()
testdata.agg({'Pressure': 'min'}).show()

#Tempearture mean and stdev values
testdata.agg({'Temp': 'mean'}).show()
testdata.agg({'Temp': 'stddev'}).show()
testdata.agg({'Temp': 'max'}).show()
testdata.agg({'Temp': 'min'}).show()

veassembler=VectorAssembler(inputCols=["Pressure", "Temp"], outputCol="features")
unltest_df = veassembler.transform(testdata).select("Pressure","Temp","features")
unltest_df.show()
print(unltest_df.count())

outputdf = model.transform(unltest_df.select("features"))
print(outputdf.count())
outputdf.show()

evaluator = ClusteringEvaluator()
silhouette = evaluator.evaluate(outputdf)
print("Silhouette with squared euclidean distance = " + str(silhouette))

#loading labeled test data
labtestdata=spark.read.format("csv").option("inferschema","true").option("header","false").load("hdfs:///hw3/test_labeled.csv",seperator=",")   
labtestdata=labtestdata.selectExpr("_c0 as Pressure", "_c1 as Temp","_c2 as label")
labtestdata=labtestdata.dropna(subset=("Pressure","Temp"),how="all")
labtestdata=labtestdata.filter(labtestdata["Pressure"]>0)

labtestdata.show()
print(labtestdata.count())

#predicting on unlabelled test data and comparing results with the labelled test data for accuracy and evaluation metrics.

output_df=outputdf.select("prediction")
lab_testdata=labtestdata.select("label")
tdf1 = output_df.withColumn("id1", monotonically_increasing_id())

tdf2 = lab_testdata.withColumn("id2", monotonically_increasing_id())

fin_df=tdf2.join(tdf1,tdf1["id1"]==tdf2["id2"],"inner").drop("id1","id2")

fin_df=fin_df.dropna(subset=("label","prediction"),how="all")

fin_df.show()

metrics1 = MulticlassMetrics(fin_df.rdd.map(lambda x: tuple(map(float, x))))
cm1 = metrics1.confusionMatrix().toArray()
labels = [int(l) for l in metrics1.call('labels')]
confusion_matrix_test = pd.DataFrame(cm1 , index=labels, columns=labels)
print(confusion_matrix_test)


precision = metrics1.precision()
recall = metrics1.recall()
f1Score = metrics1.fMeasure()
print("Summary Stats")
print("Precision = %s" % precision)
print("Recall = %s" % recall)
print("F1 Score = %s" % f1Score)

for label in sorted(labels):
    print("Class %s precision = %s" % (label, metrics1.precision(label)))
    print("Class %s recall = %s" % (label, metrics1.recall(label)))
    print("Class %s F1 Measure = %s" % (label, metrics1.fMeasure(float(label),beta=1.0)))

print("Weighted recall = %s" % metrics1.weightedRecall)
print("Weighted precision = %s" % metrics1.weightedPrecision)
print("Weighted F(1) Score = %s" % metrics1.weightedFMeasure())
print("Weighted F(0.5) Score = %s" % metrics1.weightedFMeasure(beta=0.5))
print("Weighted false positive rate = %s" % metrics1.weightedFalsePositiveRate)


