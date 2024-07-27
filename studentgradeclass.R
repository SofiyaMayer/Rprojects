df <- read.csv("studentperfomance.csv") # loading dataset

View(data)
View(df)
df <- df[, -c(1,14)]
df$GradeClass <- as.factor(df$GradeClass)
head(df, 5)
tail(df, 5)

sum(is.na(df)) # checking zero values

# Summary information about the data
library(skimr)
skim(df)

library(caret)
set.seed(123) # For reproducible model

# Data splitting
TrainingIndex <- createDataPartition(df$GradeClass, p = 0.8, list = FALSE)
TrainingSet <- df[TrainingIndex,]
TestingSet <- df[-TrainingIndex,]

# SVM Classification model for Grade Class:
Model_svm <- train(GradeClass ~ ., data = TrainingSet,
                   method = "svmPoly",
                   na.action = na.omit,
                   trControl = trainControl(method = "none"),
                   tuneGrid = data.frame(degree=1, scale=1, C=1)
)

Model_svm.cv <- train(GradeClass ~ ., data = TrainingSet,
                      method = "svmPoly",
                      na.action = na.omit,
                      trControl = trainControl(method = "cv", number = 10),
                      tuneGrid = data.frame(degree=1, scale=1, C=1)
)

# Applying the model:
Model_svm.training <- predict(Model_svm, TrainingSet)
Model_svm.testing <- predict(Model_svm, TestingSet)
Model_svm.cv <- predict(Model_svm.cv, TrainingSet)

# Check the unique levels
unique(TrainingSet$GradeClass)
unique(Model_svm.training)

print(confusionMatrix(Model_svm.training, TrainingSet$GradeClass))
print(confusionMatrix(Model_svm.testing, TestingSet$GradeClass))
print(confusionMatrix(Model_svm.cv, TrainingSet$GradeClass))       

# Feature Importance SVM:
Importance_svm <- varImp(Model_svm)
plot(Importance_svm)

# Random Forest model for GradeClass

Model_rf <- train(
  GradeClass ~ .,
  data = TrainingSet,
  method = "rf",
  preProcess = NULL,
  weights = NULL,
  metric = "Accuracy", # Set to "Accuracy" for classification
  trControl = trainControl(method = "none"),
  tuneGrid = NULL
)

# Applying Random Forest model
Model_rf.training <- predict(Model_rf, TrainingSet)
Model_rf.testing <- predict(Model_rf, TestingSet)

# Confusion Matrix:
print(confusionMatrix(Model_rf.training, TrainingSet$GradeClass))
print(confusionMatrix(Model_rf.testing, TestingSet$GradeClass))

# Feature Importance RF:
Importance_rf <- varImp(Model_rf)
plot(Importance_rf)
