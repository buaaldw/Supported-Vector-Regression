###Load required libraries
require(e1071)

###Load dataset
dataset <- read.csv("campus.csv")

##divide dataset into train and test
###set percentage for train set
perct <- 0.75
trainIndex <- round(length(dataset[,1])*perct)
train <- dataset [1:trainIndex,]
test <- dataset [-(1:trainIndex),]

##Create and fit the model with desired epsilon, kernel and Cost values
###in this example the value we want to predict is y and it is trained using all the rest of values
print("Fitting SVR model this could take some time...")
modelsvm <- svm(y ~ . ,data=train,cost=1000,kernel="radial",epsilon=0.5)
predictedsvm <- predict(modelsvm, newdata=test)

X11()  ####to show plots when running Rscript
plot(1:length(test[,1]),test$y,pch=16,ylab="Predicted value",xlab="")
points(1:length(test[,1]),predictedsvm,type="l",col="red")

#next two lines should be used when running Rscript to mantain the plot window opened
message("Press Return To Close window and continue")
invisible(readLines("stdin", n=1))


####In case we want to perform a search to find optimum values for epsilon and C
####we can perform a grid search

#tuneResult <- tune(svm, Y ~ .,  data = train,
#                   ranges = list(epsilon = seq(0,1,0.1), cost = 2^(2:9))
#)
#print(tuneResult)
## Draw the tuning graph
#plot(tuneResult)
