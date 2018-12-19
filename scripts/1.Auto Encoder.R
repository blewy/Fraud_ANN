 
load('data.model.all.rda')  %>% print()


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

# alternative
#https://blog.keras.io/building-autoencoders-in-keras.html

input_data= layer_input(shape=ncol(x_train))
encoded = layer_dense(units =20, activation='relu')(input_data)
encoded = layer_dense(units =15, activation='relu')(encoded)
encoded = layer_dense(units =10, activation='relu')(encoded)

decoded = layer_dense(units =10, activation='relu')(encoded)
decoded = layer_dense(units =15, activation='relu')(decoded)
decoded = layer_dense(units = ncol(x_train))(decoded)

autoencoder = keras_model(inputs = input_data, outputs =decoded)
supervised = keras_model(inputs=input_data, output=encoded)
autoencoder %>% compile(loss = "mean_squared_error", 
                        optimizer = "adam")

supervised %>% compile(loss = "mean_squared_error", 
                        optimizer = "adam")
#supervised.compile(...)

summary(autoencoder)

checkpoint <- callback_model_checkpoint(
  filepath = "models/autoencoder.model.hdf5", 
  save_best_only = TRUE, 
  period = 1,
  verbose = 1
)

early_stopping <- callback_early_stopping(patience = 5)

autoencoder %>% fit(
  x = x_train[y_train == 0,], 
  y = x_train[y_train == 0,], 
  epochs = 100, 
  batch_size = 128,
  shuffle=TRUE,
  validation_data = list(x_evaluate[y_evaluate == 0,], x_evaluate[y_evaluate == 0,]), 
  callbacks = list(checkpoint, early_stopping)
)


loss <- evaluate(autoencoder, x = x_test[y_test == 0,], y = x_test[y_test == 0,])
loss

supervised %>% save_model_hdf5("ancoder.model.hdf5")

# library(keras)
# model <- keras_model_sequential()
# model %>%
#   layer_dense(units = 15, activation = "tanh", input_shape = ncol(x_train)) %>%
#   layer_dense(units = 10, activation = "tanh") %>%
#   layer_dense(units = 15, activation = "tanh") %>%
#   layer_dense(units = ncol(x_train))
# 
# summary(model)
# 
# 
# 
# model %>% compile(
#   loss = "mean_squared_error", 
#   optimizer = "adam"
# )
# 
# 
# 
# checkpoint <- callback_model_checkpoint(
#   filepath = "model.hdf5", 
#   save_best_only = TRUE, 
#   period = 1,
#   verbose = 1
# )
# 
# early_stopping <- callback_early_stopping(patience = 5)
# 
# model %>% fit(
#   x = x_train[y_train == "No",], 
#   y = x_train[y_train == "No",], 
#   epochs = 100, 
#   batch_size = 32,
#   validation_data = list(x_test[y_test == "No",], x_test[== "No",]), 
#   callbacks = list(checkpoint, early_stopping)
# )
# 
# loss <- evaluate(model, x = x_test[y_test == "No",], y = x_test[y_test == "No",])
# loss
