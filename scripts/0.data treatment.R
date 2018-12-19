library(readr)
library(Amelia)
library(tidyverse)
library(caret)
library(corrplot)
library(keras)
creditcard <- read_csv("data/creditcard.csv")
View(creditcard)

source("scripts/aux_functions.R")

# Load
prop.table(table(creditcard$Class))

creditcard$Class<- factor(creditcard$Class, labels = c("No", "Yes"))

#save in separate vector
target.class.data<- factor(creditcard$Class, labels = c("No", "Yes"))
target.data <- if_else(target.class.data=="Yes",1,0)


prop.table(table(target.data))

#Take a look
creditcard[sample(nrow(creditcard),6),]

#structure
str(creditcard)

#dimension
dim(creditcard)

# Classe of the variable 
sapply(creditcard, class)

# Setting the variable ROles 

(vars <- names(creditcard))
target <- "TARGET"
id <- "Time"

# Lets sett-up the variable to ignore

ignore <- id

#lookinh for variable thar mostly serve as id
(ids <- which(sapply(creditcard, function(x) length(unique(x))) == nrow(creditcard)))
#only the ID, that good

ignore <- union(ignore, names(ids))

# All Missing (missing value count - mvc)
mvc <- sapply(creditcard[vars], function(x) sum(is.na(x)))
#all values missing
mvn <- names(which(mvc == nrow(creditcard)))
mvn
#not one 
ignore <- union(ignore, mvn)

#Many Missing 
mvn <- names(which(mvc >= 0.7*nrow(creditcard)))
mvn #not one
ignore <- union(ignore, mvn)

#Too Many Levels
factors <- which(sapply(creditcard[vars], is.factor))
lvls <- sapply(factors, function(x) length(levels(creditcard[[x]])))
(many <- names(which(lvls > 20)))

ignore <- union(ignore, many)
#constants variables

(constants <- names(which(sapply(creditcard[vars], function(x) all(x == x[1L])))))
ignore <- union(ignore, constants)

##### Removing identical features
features_pair <- combn(names(creditcard), 2, simplify = F)
toRemove <- c()
for(pair in features_pair) {
  f1 <- pair[1]
  f2 <- pair[2]
  
  if (!(f1 %in% toRemove) & !(f2 %in% toRemove)) {
    if (all(creditcard[[f1]] == creditcard[[f2]])) {
      # cat(f1, "and", f2, "are equals.\n")
      toRemove <- c(toRemove, f2)
    }
  }
}

ignore <- union(ignore, toRemove)


### Remove features with less than 10% of non-zero entries
zero.rate <- sapply(creditcard[vars], function(dt.col) {
  sum(dt.col == 0)/length(dt.col)
})

(non_info <- vars[zero.rate > 0.9])

ignore <- union(ignore, non_info)

#nzv <- nearZeroVar(creditcard, saveMetrics= FALSE)
#to.remove<-names(creditcard)[nzv]
#ignore <- union(ignore, to.remove)

correlations <- cor(creditcard[,-c(1,ncol(creditcard))])
corrplot(correlations, method="square", order="hclust")

#Find high correlated variables
summary(correlations[upper.tri(correlations)])
highlyCorDescr <- findCorrelation(correlations, cutoff = .85)

to.remove<- c(names(train_num.inputed)[highlyCorDescr])
ignore <- union(ignore, to.remove)


vars_final <- setdiff(vars, ignore)

data.model<-creditcard[,vars_final]
# Feature engeniering

## replace with most common
#data.test$var3[data.test$var3==-999999] <- 2
#data.test$var38 <- log(data.test$var38)

save(id, data.model, file='data.model.rda') 

df_train <- data.model %>% filter(row_number(Time) <= 200000) %>% select(-Time,-Class)
df_train.target_class <- data.model %>% filter(row_number(Time) <= 200000) %>% select(Class)
df_train.target <- if_else(df_train.target_class=="Yes",1,0)

df_test <- data.model %>% filter(row_number(Time) > 200000) %>% select(-Time,-Class)
df_test.target_class  <- data.model %>% filter(row_number(Time) > 200000) %>% select(Class)
df_test.target <- if_else(df_test.target_class=="Yes",1,0)

set.seed(934)
split <- createDataPartition(df_train.target_class$Class, p = 1/3)[[1]]
df_evaluation  <- df_train[split,]
df_evaluation_target.class <- df_train.target_class$Class[split]
df_evaluation_target <- df_train.target[split]

df_train     <- df_train[-split,]
df_train.target_class <- df_train.target_class$Class[-split]
df_train.target <- df_train.target[-split]

save(id, 
     df_train,df_train.target_class,df_train.target,
     df_evaluation,df_evaluation_target.class,df_evaluation_target,
     df_test,df_test.target_class,df_test.target,
     file='data.model.all.rda') 
