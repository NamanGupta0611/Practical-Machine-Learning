---
title: "Practical ML - Data Science specialization"
author: "Naman Gupta"
date: '`r Sys.Date()`'
output: 
  html_notebook:
    theme: cosmo
    toc: yes
    toc_float: yes
    fig_width: 5
    fig_heigth: 5
    code_folding: hide
  html_document: default
---

## Introduction

This document is submitted as a Assignment writeup for the Coursera's [Practical Machine Learning] Course.

## Background

Subjects were asked to perform barbell lifts correctly and incorrectly in 5 different ways.
* Exactly according to the specification (Class A)
* Throwing the elbows to the front (Class B) - mistake
* Lifting the dumbbell only halfway (Class C) - mistake
* Lowering the dumbbell only halfway (Class D) - mistake
* Throwing the hips to the front (Class E) - mistake

Accelerometers were located on
1. belt
2. forearm
3. arm

## Task

Create a report describing
* how you built your model,
* how you used cross validation
* what you think the expected out of sample error is
* why you made the choices you did

## Overview

The model building workflow adopted for this task follows the pattern outlined in lectures:

> question .. input .. features .. algorithm .. predict .. evaltuation

Cross Validation has been used as a method for the trainControl function with 4 folds used.

The out of sample error was found to be 0.0037% when the model was applied to the test data derived from the training set.

Choices made at each step are described in the workflow below.

## Setup

Due to size of the training sample (19622 observations and up to 60 variables), parallel processing was selected for model development

Loading the required libraries

```{r}
suppressWarnings(suppressMessages(library(caret)))
suppressWarnings(suppressMessages(library(randomForest)))
suppressWarnings(suppressMessages(library(e1071)))
```

## Question

Create a model to predict the manner in which the subjects did the exercise using the accelerometer data as predictors.
The outcome to be predicted is the “classe” variable.

## Input

### Download Source data

```{r}
trainingFilename   <- 'pml-training.csv'
testingFilename       <- 'pml-testing.csv'
trainingUrl        <- 'https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv'
testingUrl            <- 'https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv'
download.file(trainingUrl, trainingFilename)
download.file(testingUrl, testingFilename)
```

### Data reading and cleansing

On inspection in Excel, found NA,#DIV/0! and blank values in the data. These are not valid observed values, so remove with na.strings parameter.

```{r}
training.df <- read.csv(trainingFilename, na.strings = c("NA","","#DIV/0"))
testing.df <- read.csv(testingFilename, na.strings = c("NA","","#DIV/0"))

training.df <- training.df[, colSums(is.na(training.df)) == 0]
testing.df <- testing.df[, colSums(is.na(testing.df)) == 0]

dim(training.df)
dim(testing.df)
```

## Features

### Reduce the number of variables

Remove the non-predictors from the training set. This includes the index, subject name, time and window variables.

```{r}
training.df <- training.df[, -c(1:7)]
testing.df <- testing.df[, -c(1:7)]

dim(training.df)
dim(testing.df)
```

### Check for near zero variables in the training dataset

```{r}
training.nzv <- nzv(training.df[, -ncol(training.df)], saveMetrics = TRUE)
```

None found so display and count variables submitted for the train function

```{r}
rownames(training.nzv)
```

```{r}
dim(training.nzv)
```

## Algorithm

### Partitioning the training data into training set and tresting/validation set

```{r}
inTrain <- createDataPartition(y = training.df$classe, p = 0.6, list = FALSE)
training <- training.df[inTrain, ]
testing <- training.df[-inTrain, ]
dim(training)
dim(testing)
```

### Construct the model using cross validation or reload using the cached model

Cross Validation achieved with trainControl method set to “cv”

```{r}
modelFileName <- "myModel.Rdata"
if(!file.exists(modelFileName)) {
  
  # Parallel Cores
  # require(parallel)
  library(doParallel)
  ncores <- makeCluster(detectCores() - 1)
  registerDoParallel(cores = ncores)
  getDoParWorkers()
  
  # use Random Forest method with Cross Validation, 4 folds
  myModel <- train(classe ~ .,
                   data = training,
                   method = "rf",
                   metric = "Accuracy", # categorical outcome variable so choose accuracy
                   preProcess = c("center", "scale"), # attempt to improve accuracy by normalising
                   trControl = trainControl(method = "cv",
                                            number = 4,
                                            p = 0.6,
                                            allowParallel = TRUE))
  save(myModel, file = "myModel.Rdata")
  stopCluster(ncores)
}else{
  # Use cached Model
  load(file = modelFileName, verbose = TRUE)
}
```

```{r}
print(myModel, digits = 4)
```

## Predict

Predicting the activity performed using the training file derived test subset

```{r}
predTest <- predict(myModel, newdata = testing)
```

## Evaluation

### Test

Check the accuracy of the model by comparing the predictions to the actual results

```{r}
confusionMatrix(predTest, as.factor(testing$classe))
```

### Out of Sample Error

The out-of-sample error of 0.0037 or 0.37%.

Accuracy is very high, at 0.9963, and this figure lies within the 95% confidence interval.

### Final Model data and important predictors in the model

```{r}
myModel$finalModel
```

```{r}
varImp(myModel)
```

27 variables were tried at each split and the reported OOB Estimated Error is a low 0.83%.

Overall we have sufficient confidence in the prediction model to predict classe for the 20 quiz/test case

### Validation/Quiz

The accuracy of the model by predicting with the Validation/Quiz set supplied in the test file.

```{r}
print(predict(myModel, newdata = testing.df))
```