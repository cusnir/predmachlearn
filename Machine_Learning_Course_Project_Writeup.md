# Human Activity Recognition: Predicting exercise type manner
Cușnir Andrei  
2015 November, 20  

## Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).  

## Executive Summary

The goal of this project is to predict the manner in which the 6 participants did the exercise. the "classe" variable in the training set will be used for this purpose.
Also will use choosed prediction model to predict 20 different test cases.



```r
library(caret)
library(randomForest)
library(rpart)
```


```r
# download training data file from url
if (!file.exists("pml-training.csv")) {
    download.file("http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", destfile = "pml-training.csv", method="curl")
}
# download testing data file from url
if (!file.exists("pml-testing.csv")) {
    download.file("http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", destfile = "pml-testing.csv", method="curl")
}
```

Reading both training and testing data sets  

```r
training<-read.csv("pml-training.csv", na.strings=c("NA","#DIV/0!"))
testing<-read.csv("pml-testing.csv", na.strings=c("NA","#DIV/0!"))
```

## Data exploring  


```r
dim(training)
```

```
## [1] 19622   160
```

```r
table(training$classe)
```

```
## 
##    A    B    C    D    E 
## 5580 3797 3422 3216 3607
```

Making sure we are not using records with NA which can create noise in our model.
Some variables are used as identifiers so we are removing these too
Those variables are "X", "user_name", and all the time related variables, such as "raw_timestamp_part_1", etc...   


```r
NA_Count <- sapply(1:dim(training)[2], function(x)sum(is.na(training[,x])))
NA_list <- which(NA_Count>0)
# making sure about column names of identifiers we will remove from training set
colnames(training[,c(1:7)])
```

```
## [1] "X"                    "user_name"            "raw_timestamp_part_1"
## [4] "raw_timestamp_part_2" "cvtd_timestamp"       "new_window"          
## [7] "num_window"
```

```r
training <- training[,-NA_list]
training <- training[,-c(1:7)]
training$classe = factor(training$classe)
```


## Moddeling with Cross-validation

The problem presented here is a classification problem.  
We used caret package classification methods: classification tree algorithm and random forest. 
Also a 3-fold validation was done using trainControl function.

For speeding up training will use here multicore abilities of train function, 
for this we have to load doMC library and set cores number as per present number of cores in current computer


```r
set.seed(2434)

library(doMC)
registerDoMC(cores = 4)
## All subsequent models are then run in parallel

crval <- trainControl(method="cv", number=3, allowParallel=TRUE, verboseIter=TRUE)
model.rf <- train(classe~., data=training, method="rf", trControl=crval, allowParallel = TRUE)
```

```
## Aggregating results
## Selecting tuning parameters
## Fitting mtry = 27 on full training set
```

```r
model.tree <- train(classe~., data=training, method="rpart", trControl=crval)
```

```
## Aggregating results
## Selecting tuning parameters
## Fitting cp = 0.0357 on full training set
```

The performance of these two models on the training dataset was:  

```r
predict.rf <- predict(model.rf, training)
predict.tree <- predict(model.tree, training)

table(predict.rf, training$classe)
```

```
##           
## predict.rf    A    B    C    D    E
##          A 5580    0    0    0    0
##          B    0 3797    0    0    0
##          C    0    0 3422    0    0
##          D    0    0    0 3216    0
##          E    0    0    0    0 3607
```

```r
table(predict.tree, training$classe)
```

```
##             
## predict.tree    A    B    C    D    E
##            A 5080 1581 1587 1449  524
##            B   81 1286  108  568  486
##            C  405  930 1727 1199  966
##            D    0    0    0    0    0
##            E   14    0    0    0 1631
```
And the performance of these two models on the testing dataset was:  

```r
predict.rf <- predict(model.rf, testing)
predict.tree <- predict(model.tree, testing)
table(predict.rf, predict.tree)
```

```
##           predict.tree
## predict.rf A B C D E
##          A 7 0 0 0 0
##          B 3 0 5 0 0
##          C 0 0 1 0 0
##          D 0 0 1 0 0
##          E 1 0 2 0 0
```
From the results, we can see that the random forest model has the best accuracy for testing dataset.  

## Predictions
Random forest model was used on testing dataset with 20 raws to predict new data.
pml_write_files function was used to create response files.

```r
answers <- predict(model.rf, testing)
answers
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

```r
pml_write_files <- function(x){
    if (!file.exists("data")) {dir.create("data")}
    n <- length(x)
    for(i in 1:n){
        filename <- paste0("data/", "problem_id_",i,".txt")
        write.table(x[i], file <- filename, quote=FALSE, row.names=FALSE, col.names=FALSE)
    }
}
pml_write_files(answers)
```
## Conclusions

From the given training dataset from different measuring sensors it is possible to create quite acurately a model 
which will predict how well a person is exercising.
