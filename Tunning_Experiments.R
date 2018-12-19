library(tfruns)
FLAGS <- flags(
 flag_numeric("dropout1", 0.4),
 flag_integer('units1', 20L)
 )
training_run("scripts/2.predict_fraud.R")
training_run('scripts/2.predict_fraud.R', flags = list(dropout1 = 0.4,units1=20L))

for (dropout1 in c(0.1, 0.2, 0.3))
  training_run('scripts/2.predict_fraud.R', flags = list(dropout1 = dropout1))

# 
# # define flags and parse flag values from flags.yml and the command line
# FLAGS <- flags(
#   flag_numeric('learning_rate', 0.01, 'Initial learning rate.'),
#   flag_integer('max_steps', 5000, 'Number of steps to run trainer.'),
#   flag_string('data_dir', 'MNIST-data', 'Directory for training data'),
#   flag_boolean('fake_data', FALSE, 'If true, use fake data for testing')
# )
# # }

runs <- tuning_run("scripts/2.predict_fraud.R", flags = list(
  dropout1 = c(0.2, 0.4),
  units1=20L
))

# find the best evaluation accuracy
runs[order(runs$metric_loss, decreasing = TRUE), ]

View(runs)


runs <- tuning_run("scripts/2.predict_fraud.R", sample = 0.5, flags = list(
  dropout1 = c(0.2, 0.3, 0.4),
  units1 = c(5, 10, 20),
  dropout2= c(0.2, 0.3, 0.4),
  units2= c(5, 10, 20)
))


# find the best evaluation accuracy
runs[order(runs$metric_loss, decreasing = TRUE), ]

View(runs)

view_run("runs/2018-12-15T21-20-17Z")
ls_runs(order = eval_acc, runs_dir = "dropout_tuning")


training_run('scripts/2.predict_fraud.R', flags = list(dropout1 = 0.2,units1=10L))
