# Fraud_ANN
Research into using ANN (Autoencoder + MLP) por fraud detection

In this small project I tried to build a predictive model usigng a two stage model:

  - Build a autoencoder that encapsulates the information about the majority of transations (not fraud)
  - Use the autoencoder as first layer for a MLP to predict fraud.
  
 As at now I only used ROse resample for building more balanced training data set. I will add more later
