
library(readr)
library(Amelia)
library(tidyverse)
library(caret)
library(corrplot)
library(keras)

auc.estimate <- function(true,prediction){
  require(pROC)
  return(auc(roc(true, prediction)))
}

load('data.model.all.rda')  %>%  print()
## Area under the curve: 0.825
# roc_df <- data.frame(
#   TPR=rev(roc_obj$sensitivities), 
#   FPR=rev(1 - roc_obj$specificities), 
#   labels=roc_obj$response, 
#   scores=roc_obj$predictor)

encoder<- load_model_hdf5("ancoder.model.hdf5", custom_objects = NULL, compile = TRUE)

summary(encoder)


library(purrr)
#' Gets descriptive statistics for every variable in the dataset.
get_desc <- function(x) {
  map(x, ~list(
    min = min(.x),
    max = max(.x),
    mean = mean(.x),
    sd = sd(.x)
  ))
} 

#' Given a dataset and normalization constants it will create a min-max normalized
#' version of the dataset.
normalization_minmax <- function(x, desc) {
  map2_dfc(x, desc, ~(.x - .y$min)/(.y$max - .y$min))
}

desc <- df_train %>% 
  get_desc()

x_train <- df_train %>%
  normalization_minmax(desc) %>%
  as.matrix()

x_test <- df_test %>%
  normalization_minmax(desc) %>%
  as.matrix()

x_evaluate <- df_evaluation %>%
  normalization_minmax(desc) %>%
  as.matrix()

y_train <- df_train.target
y_test <- df_test.target
y_evaluate <- df_evaluation_target


# create the base pre-trained model
base_model <- encoder

FLAGS <- flags(
  flag_numeric("dropout1", 0.2),
  flag_integer("units1", 20L),
  flag_numeric("dropout2", 0.4),
  flag_integer("units2", 20L)
)

# add our custom layers
predictions <- base_model$output %>% 
  layer_dense(FLAGS$units1, activation = 'relu') %>% 
  layer_dropout(FLAGS$dropout1) %>% 
  layer_dense(FLAGS$units2, activation = 'relu') %>% 
  layer_dropout(FLAGS$dropout2) %>% 
  layer_dense(units = 1, activation = 'sigmoid')

# this is the model we will train
model <- keras_model(inputs = base_model$input, outputs = predictions)
summary(model)
# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
freeze_weights(base_model)

# compile the model (should be done *after* setting layers to non-trainable)
model %>% compile(optimizer = 'adam', loss = 'binary_crossentropy')

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
layers <- base_model$layers
for (i in 1:length(layers))
  cat(i, layers[[i]]$name, "\n")

# # we chose to train the top 2 inception blocks, i.e. we will freeze
# # the first 172 layers and unfreeze the rest:
# freeze_weights(base_model, from = 1, to = 172)
# unfreeze_weights(base_model, from = 173)
# 
# # we need to recompile the model for these modifications to take effect
# # we use SGD with a low learning rate
# model %>% compile(
#   optimizer = optimizer_sgd(lr = 0.0001, momentum = 0.9), 
#   loss = 'binary_crossentropy'
# )

dir.create("my_log_dir")

checkpoint <- callback_model_checkpoint(
  filepath = "models/final.fraud.predict.model.hdf5", 
  save_best_only = TRUE, 
  period = 1,
  verbose = 1
)

early_stopping <- callback_early_stopping(patience = 10)

tensorboard <- callback_tensorboard(
  log_dir = "my_log_dir",
  histogram_freq = 1
)

#y_train <- if_else(y_train=="Yes",1,0)
#y_test <- if_else(y_test=="Yes",1,0)

library(ROSE)
rose.train <- as.data.frame(cbind(x_train,y_train))
#View(rose.train)
rose.train$Class <- as.factor(rose.train$y_train)
rose.train$y_train <- NULL
down_train <- ROSE(Class ~ ., data  = rose.train)
table(down_train$data$Class)


x_train.ub<-down_train$data %>% select(-Class) %>%  as.matrix
y_train.ub<-ifelse(down_train$data$Class=="0",0,1)


#tensorboard("my_log_dir")
history <-model %>% fit(
  x = x_train.ub, 
  y = y_train.ub, 
  epochs = 100, 
  batch_size = 1024,
  shuffle=TRUE,
  validation_data = list(x_evaluate,y_evaluate), 
  callbacks = list(checkpoint, early_stopping)
)

plot(history)

loss <- evaluate(model, x = x_test, y = y_test)
loss
predited <- predict(model, x = x_test)
predited

threshold =0.5
library(forcats)
# estimates_keras_tbl <- tibble(
#   truth      = as.factor(y_test) %>% fct_recode(0 = "no", 1 = "yes"),
#   estimate   = as.factor(ifelse(predited[,1]>threshold,1,0)) %>% fct_recode(0 = "no", 1 = "yes"),
#   class_prob = predited[,1]
# )

estimates_keras_tbl <- tibble(
  truth      = as.factor(y_test) ,
  estimate   = as.factor(ifelse(predited[,1]>threshold,1,0)),
  class_prob = predited[,1]
)


library(Metrics)

predictions <- predict(model, x = x_test)
dim(predictions)
auc.estimate(y_test,predictions[,1] )
table(estimates_keras_tbl$truth,estimates_keras_tbl$estimate)

confusionMatrix(estimates_keras_tbl$estimate,estimates_keras_tbl$truth)

#Lift Curve
lift_obj <- lift(truth ~ class_prob, data = estimates_keras_tbl)
plot(lift_obj, values = 60, auto.key = list(columns = 3,
                                            lines = TRUE,
                                            points = FALSE))


#Calibration
cal_obj <- calibration(truth ~ class_prob, 
                       data = estimates_keras_tbl,
                       cuts = 13)
plot(cal_obj, type = "l", auto.key = list(columns = 1,
                                          lines = TRUE,
                                          points = FALSE))

library(rattle)
riskchart(estimates_keras_tbl$class_prob,estimates_keras_tbl$truth,x_test[,"Amount"] )
colnames(x_evaluate)
