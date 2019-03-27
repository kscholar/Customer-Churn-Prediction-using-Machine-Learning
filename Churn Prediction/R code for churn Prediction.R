# removing all the previous variables #
rm(list=ls())
#Importing the all necessary library#
library('ggplot2') 
library('corrgram') 
library('corrplot') 
library('caret') 
library('class') 
library('e1071') 
library('rpart') 
library('mlr')
library('grid') 
library('DMwR') 
library('irace') 
library('usdm')
library('randomForest')
## Reading train and test csv file and settting working directory to the file location ##
setwd("D:/Dataset/project Dataset")
train_set = read.csv("Train_data.csv")
test_set = read.csv("Test_data.csv")

################ Data Preprocessing ################ 

# Checking columns name, 
colnames(train_set)     
# checking datatypes of all columns
str(train_set)
# Checking numerical statistics of numerical columns (Five point summary + mean of all column)
summary(train_set)

### Checking categorical variable ###
# unique values in each category
categorical = c('area.code','international.plan', 'voice.mail.plan','Churn')
lapply(train_set[,c('state', categorical)], function(feat) length(unique(feat)))

# counting of each unique values in categorical columns
lapply(train_set[,categorical], function(feature) table(feature))



################ Missing value Analysis ################ 


# Storing all the missing values in separate data frame #
missing_value = data.frame(lapply(train_set, function(feat) sum(is.na(feat))))


################ outlier Analysis ################ 

# removing phone number column and changing area code to category
train_set$phone.number = NULL
train_set$area.code = as.factor(train_set$area.code)

test_set$phone.number = NULL
test_set$area.code = as.factor(test_set$area.code)

# taking out list of name of numerical columns in dataset
numerical = colnames(Filter(is.numeric, train_set))


# Creating a function to plot boxplot #
box_plot = function(column, dataset){
  ggplot(aes_string(x = 'Churn', y = column, fill = 'Churn'),
         data = dataset)+
    stat_boxplot(geom = 'errorbar', width = 0.5)+
    geom_boxplot(outlier.size = 2, outlier.shape = 18)+
    theme(legend.position = 'bottom')+
    labs(y = gsub('\\.', ' ', column), x = "Churn")+
    ggtitle(paste(" Box Plot :",gsub('\\.', ' ', column)))
}

# Creating a function to plot histogram #
hist_plot = function(column, dataset){
  ggplot(aes_string(column), data = dataset)+
    geom_histogram(aes(y=..density..), fill = 'skyblue2')+
    geom_density()+
    labs(x = gsub('\\.', ' ', column))+
    ggtitle(paste(" Histogram :",gsub('\\.', ' ', column)))
}


# Storing all the boxplot in a list
all_box_plots = lapply(numerical,box_plot, dataset = train_set)

# Storing all the Histograms in a list
all_hist_plots = lapply(numerical,hist_plot, dataset = train_set)

# Plotting Boxplot and histogram #
grid_plot = function(f, s, t){
  gridExtra::grid.arrange(all_box_plots[[f]],all_box_plots[[s]],all_box_plots[[t]],
                          all_hist_plots[[f]],all_hist_plots[[s]],all_hist_plots[[t]],ncol=3,nrow=2)
}

# plotting for day's minute, call and charges
grid_plot(3,4,5)

# plotting for evening's minute, call and charges
grid_plot(6,7,8)

# plotting for night's minute, call and charges
grid_plot(9, 10, 11)

# plotting for international's minute, call and charges
grid_plot(12, 13, 14)

# plotting for account length, voice mail message and customer service calls
grid_plot(1, 2, 15)


###################### Outlier removal ######################
# Note: Considering both dataset one with outliers and other without outliers for building model
# dataset with outlier :- train_set
# dataset without outlier:- no_outlier

no_outlier = train_set

# removing numeric columns for which we will not do outlier removal process
numerical_1 = numerical[! numerical %in% c("number.vmail.messages","number.customer.service.calls")]

for (i in numerical_1){
  out_value = no_outlier[,i] [no_outlier[,i] %in% boxplot.stats(no_outlier[,i])$out]
  no_outlier = no_outlier[which(!no_outlier[,i] %in% out_value),]
}

# Plotting again distribution and boxplot after outlier removal

# calling box_plot function and storing all plots in a list  for  dataset without outliers
all_box_plots = lapply(numerical,box_plot, dataset = no_outlier)

# calling hist_plot function and storing all plots in a list dataset without outliers
all_hist_plots = lapply(numerical,hist_plot, dataset = no_outlier)

# plotting for day's minute, call and charges after outlier removal
grid_plot(3,4,5)

# plotting for evening's minute, call and charges after outlier removal
grid_plot(6,7,8)

# plotting for night's minute, call and charges after outlier removal
grid_plot(9, 10, 11)

# plotting for international's minute, call and charges after outlier removal
grid_plot(12, 13, 14)

# plotting for account length, voice mail message and customer service calls after outlier removal
grid_plot(1, 2, 15)


#################################### Feature Selection ####################################

# correlation plot for numerical feature
corrgram(train_set[,numerical], order = FALSE,
         upper.panel = panel.pie, text.panel = panel.txt,
         main = "Correlation Plot")

# heatmap for numerical features
corrplot(cor(train_set[,numerical]), method = 'color', type = 'lower')

# getting categorical column
categorical = c('state', 'area.code','international.plan', 'voice.mail.plan')

# chi-square test of independence of each category with Churn column
for(i in categorical){
  print(i)
  print(chisq.test(table(train_set$Churn, train_set[,i])))
}

# Now checking multicollinearity between international plan and voice mail plan
# by chi-sq test of independence
print(chisq.test(table(train_set$international.plan,
                       train_set$voice.mail.plan)))

# checking VIF factor for numeric columns
vif(train_set[,numerical])

# checking importance of feature in ranking using random forest
feat = randomForest(Churn ~ ., data = train_set,
                               ntree = 200, keep.forest = FALSE, importance = TRUE)
feat_df = data.frame(importance(feat, type = 1))

################################### Preprocessed Data #################################

# Dropping column state, area code as in chi-sq test these column were
# not dependent with Churn column. Dropping  total day min, total eve charge, total night charge,
# total intl charge and these columns found to be multicolinear with other columns
train_set = train_set[, -c(1,3,7,12,15,18)]
no_outlier = no_outlier[, -c(1,3,7,12,15,18)]
test_set = test_set[, -c(1,3,7,12,15,18)]


# checking VIF factor for numeric columns 
numerical = colnames(Filter(is.numeric, train_set))
vif(train_set[,numerical])

# changing levels to 0 and 1
# no = 0, yes= 1
# false. = 0, true. = 1
category = c('international.plan', 'voice.mail.plan', 'Churn')

for (i in category){
  levels(train_set[,i]) <- c(0,1)
  levels(no_outlier[,i]) <- c(0,1)
  levels(test_set[,i]) <- c(0,1)
}


######################## Building Classification models ########################

######################## K-fold CV accuracy score calculation ########################

### Function for calculating the K-fold CV accuracy ###
model.K_fold.accuracy <- function(classifier, data){
  # creating 10 folds of data
  ten_folds = createFolds(data$Churn, k = 10)
  # lapply function will result in 10 accuracy measure for each test fold
  ten_cv = lapply(ten_folds, function(fold) {
    training_fold = data[-fold, ]
    test_fold = data[fold, ]
    
    # changing data of classifier with our training folds
    classifier$data = training_fold
    # predicting on test folds
    # for logisitic regression "glm" converting probability to class
     
    if(class(classifier)[1] == "glm"){
      y_prob = predict(churn_classifier, type = 'response', newdata = test_fold[-14])
      y_pred = ifelse(y_prob>0.5, 1, 0)
    } else if(class(classifier)[1] == 'rpart'){
      y_pred = predict(churn_classifier, newdata = test_fold[-14], type ='class')
    } else{
      y_pred = predict(churn_classifier, newdata = test_fold[-14])
    }
    # creating confusion matrix 
    cm = table(test_fold[, 14], y_pred)
    # calculating accuracy correct prediction divide by all observation
    accuracy = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[1,2] + cm[2,1] + cm[2,2])
    return(accuracy)
  })
  # returning mean of all accuracy which we got from lapply function result
  return(mean(as.numeric(ten_cv)))
}

#####Function for Predicting result on test data set of a model also returning confusion matrix#####

churn.predict <- function(classifier, data){
  if(class(classifier)[1] == 'glm'){
    churn_prob <- predict(classifier, newdata = data[,-14])
    churn_prediction <- ifelse(churn_prob >= 0.5, 1, 0)
  } else if(class(classifier)[1] == 'rpart'){
    churn_prediction = predict(classifier, data[,-14], type ='class')
  } else{
    churn_prediction = predict(classifier, data[,-14])
  }
  cm = confusionMatrix(table(data$Churn, churn_prediction))
  return(cm)
}

##########################  Logistic Regression  ##########################

# logistic regression on dataset with outliers
churn_classifier <- glm(formula = Churn ~ ., family = binomial,
                        data = train_set)
cm <- churn.predict(churn_classifier, test_set)
cm

# K -fold accuracy of Logistic regression model
k_fold_accuracy <- model.K_fold.accuracy(churn_classifier, train_set)
k_fold_accuracy

# Now checking on dataset without ouliers
churn_classifier <- glm(formula = Churn ~ ., family = binomial,
                        data = no_outlier)
cm <- churn.predict(churn_classifier, test_set)
cm
# K -fold accuracy of Logistic regression model
k_fold_accuracy <- model.K_fold.accuracy(churn_classifier, no_outlier)
k_fold_accuracy


########################## K-Nearest Neighbour ##########################

# predicting on dataset with outliers 
prediction =knn(train = train_set[,-14], test = test_set[,-14],
                        cl = train_set$Churn, k = 5, prob = TRUE)
confusionMatrix(table(test_set$Churn, prediction))


# predicting on dataset with outliers 
prediction = knn(train = no_outlier[,-14], test = test_set[,-14],
                        cl = no_outlier$Churn, k = 5, prob = TRUE)
confusionMatrix(table(test_set$Churn, prediction))


########################## Naive Bayes #########################

# Building model on dataset with outliers 
churn_classifier =naiveBayes(x = train_set[,-14], y =train_set[,14])

cm = churn.predict(churn_classifier, test_set)
cm
# K -fold accuracy of Naive Bayes model
k_fold_accuracy = model.K_fold.accuracy(churn_classifier, train_set)
k_fold_accuracy


# building model on dataset without outliers 
churn_classifier = naiveBayes(x = no_outlier[,-14], y =no_outlier[,14])
cm <- churn.predict(churn_classifier, test_set)
cm
# K -fold accuracy of Naive Bayes model
k_fold_accuracy <- model.K_fold.accuracy(churn_classifier, no_outlier)
k_fold_accuracy


########################## Decision Tree ##########################

# building model on dataset with outliers 
churn_classifier <- rpart(formula = Churn ~ ., data = train_set)

cm <- churn.predict(churn_classifier, test_set)
cm
# K -fold accuracy of Decision Tree model
k_fold_accuracy <- model.K_fold.accuracy(churn_classifier, train_set)
k_fold_accuracy

# building model on dataset without outliers
classifier <- rpart(formula = Churn ~ ., data = no_outlier)
cm <- churn.predict(classifier, test_set)
cm
# K -fold accuracy of Decision Tree model
k_fold_accuracy <- model.K_fold.accuracy(classifier, no_outlier)
k_fold_accuracy


#########Hyperparameter tuning##############

#tuning decision tree for both dataset #
# train_set and no_outlier
# we will tune best model among above i.e. Decision tree and 
# for tuning we will use mlr package and its methods

tune.Decision.Tree <- function(learner, paramset, dataset){
  # creating task for train 
  train_task = makeClassifTask(data = dataset, target = 'Churn')
  
  # setting 10 fold cross validation
  cv = makeResampleDesc("CV", iters = 10)
  grid_control = makeTuneControlGrid()
  # tuning parameter
  tune_param = tuneParams(learner = learner, resampling = cv, task = train_task,
                          par.set = paramset, control = grid_control, measures = acc)
  return(tune_param)
}

# tuning decision tree classifier for train_set i.e. whole dataset
# making learner tree
learner = makeLearner("classif.rpart", predict.type = 'response')

# setting params range
param_set <- makeParamSet(
  makeIntegerParam("minsplit", lower = 10, upper = 50),
  makeIntegerParam("minbucket", lower = 5, upper = 50),
  makeNumericParam("cp", lower = 0.001, upper = 0.2)
) 

tuned_param <- tune.Decision.Tree(learner, param_set, train_set)

# building decision tree model based on tuned param with mlr package
set_param <- setHyperPars(learner, par.vals = tuned_param$x)
train_task <- makeClassifTask(data = train_set, target = 'Churn')
test_task <- makeClassifTask(data = test_set, target = 'Churn')
# training model
train_model <- train(set_param, train_task)
# predicting on test data
pred <- predict(train_model, test_task)
y_pred = pred[["data"]][["response"]]
# confusion matrix
cm = table(test_set[, 14], y_pred)
cm

