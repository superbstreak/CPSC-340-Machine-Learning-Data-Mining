clear all
load DTdata.mat

maxFold = 2;
[N,D] = size(X);
T = length(ytest);
foldN = N/maxFold;
foldT = T/maxFold;
resultTrain = zeros(15,maxFold+1);

for depth=1:15
    total = 0;
    Xpos = 1;
    for fold=1:maxFold
        foldX = X(Xpos:foldN*fold,:);
        foldy = y(Xpos:foldN*fold,:);
        model = decisionTree_InfoGain(foldX,foldy,depth);

        yhat = model.predictFunc(model,foldX);
        errorTrain = sum(yhat ~= foldy)/foldN

        fprintf('Depth: %d Fold: %d\n',depth, fold);
        resultTrain(depth,fold) = errorTrain;
        total = total + errorTrain;
        Xpos = Xpos+(foldN*fold);
    end
    resultTrain(depth,maxFold+1) = total/maxFold;
end