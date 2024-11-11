from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql import functions as F
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Step 1: Create Spark session and context
spark = SparkSession.builder \
    .appName("Lung Cancer Risk Prediction") \
    .master("local[*]") \
    .getOrCreate()

sc = spark.sparkContext

# Step 2: Load the dataset
data = spark.read.csv("/home/sat3812/Downloads/lungcancerrisk.csv", header=True, inferSchema=True)

# Step 3: Data preprocessing
categorical_columns = [
    'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE',
    'CHRONIC_DISEASE', 'FATIGUE', 'ALLERGY', 'WHEEZING',
    'ALCOHOL_CONSUMING', 'COUGHING', 'SHORTNESS_OF_BREATH',
    'SWALLOWING_DIFFICULTY', 'CHEST_PAIN', 'LUNG_CANCER'
]

# Convert categorical columns to numeric
for col in categorical_columns:
    indexer = StringIndexer(inputCol=col, outputCol=f"{col}_indexed", handleInvalid='skip')
    data = indexer.fit(data).transform(data)

# Drop original categorical columns after indexing
data = data.drop(*categorical_columns)

# Rename indexed columns back to original names
for col in categorical_columns:
    data = data.withColumnRenamed(f"{col}_indexed", col)

# Handle missing values by filling with a default value
data = data.na.fill({col: 0 for col in categorical_columns})

# Remove rows with invalid labels for LUNG_CANCER
data = data.filter(F.col('LUNG_CANCER').isNotNull())

# Convert all columns to numeric and drop non-numeric rows
for col in data.columns:
    data = data.withColumn(col, F.col(col).cast("float"))

# Drop rows with any null values that may result from conversion errors
data = data.na.drop()

# Split the data into training and testing datasets
train_data, test_data = data.randomSplit([0.8, 0.2], seed=1234)

# Step 4: Assemble features
feature_columns = ['AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 
                   'PEER_PRESSURE', 'CHRONIC_DISEASE', 'FATIGUE', 
                   'ALLERGY', 'WHEEZING', 'ALCOHOL_CONSUMING', 
                   'COUGHING', 'SHORTNESS_OF_BREATH', 
                   'SWALLOWING_DIFFICULTY', 'CHEST_PAIN']

assembler = VectorAssembler(inputCols=feature_columns, outputCol='features')
train_data = assembler.transform(train_data)
test_data = assembler.transform(test_data)

# Step 5: Define and train the Logistic Regression model
lr = LogisticRegression(featuresCol='features', labelCol='LUNG_CANCER')

# Create a grid of hyperparameters to search over
paramGrid = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.01, 0.1, 1.0]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
    .build()

# Define the evaluator
evaluator = MulticlassClassificationEvaluator(labelCol="LUNG_CANCER", predictionCol="prediction", metricName="accuracy")

# Define the cross-validator
cv = CrossValidator(estimator=lr, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5)

# Fit the cross-validator to the training data
cvModel = cv.fit(train_data)

# Path where you want to save the model
model_save_path = "/home/sat3812/lung_cancer_lr_model"

# Create the directory if it doesn't exist
os.makedirs(model_save_path, exist_ok=True)

# Save the trained model to the specified path with overwrite mode
cvModel.bestModel.write().overwrite().save(model_save_path)

# Step 6: Load the trained model
lr_model = LogisticRegressionModel.load(model_save_path)

# Step 7: Make predictions on the test data
predictions = lr_model.transform(test_data)

# Step 8: Show predictions
predictions.select("features", "prediction", "LUNG_CANCER").show()

# Step 9: Evaluate the model
accuracy = evaluator.evaluate(predictions)
precision_evaluator = MulticlassClassificationEvaluator(labelCol="LUNG_CANCER", predictionCol="prediction", metricName="weightedPrecision")
recall_evaluator = MulticlassClassificationEvaluator(labelCol="LUNG_CANCER", predictionCol="prediction", metricName="weightedRecall")
f1_evaluator = MulticlassClassificationEvaluator(labelCol="LUNG_CANCER", predictionCol="prediction", metricName="f1")

precision = precision_evaluator.evaluate(predictions)
recall = recall_evaluator.evaluate(predictions)
f1_score = f1_evaluator.evaluate(predictions)

# Print the evaluation results
print(f"Test Accuracy = {accuracy:.4f}")
print(f"Precision = {precision:.4f}")
print(f"Recall = {recall:.4f}")
print(f"F1 Score = {f1_score:.4f}")

# Step 10: Create a heatmap for correlation
# Convert Spark DataFrame to Pandas DataFrame for correlation heatmap
pandas_df = data.toPandas()

# Calculate the correlation matrix
correlation_matrix = pandas_df.corr()

# Generate the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Heatmap')
plt.show()

# Step 11: Stop Spark session
spark.stop()
