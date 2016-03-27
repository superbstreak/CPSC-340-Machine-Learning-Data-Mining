clear all
load DTdata.mat

[N,D] = size(X);
T = length(ytest);
depth = 5;

model = decisionTree_InfoGain(X,y,depth);
yhat = model.predictFunc(model,X);
errorTrain = sum(yhat ~= y)/N
yhat = model.predictFunc(model,Xtest);
errorTest = sum(yhat ~= ytest)/T
