clear all
load DTdata.mat

maxFold = 2;
[N,D] = size(X);
foldN = N/maxFold;
resultTrain = zeros(15,maxFold*2+1);
foldXa = X(1:2500,:);
foldXb = X(2501:5000,:);
foldya = y(1:2500,:);
foldyb = y(2501:5000,:);

for depth=1:15
    % Train with 1 - 2500 and test with 2501 - 5000
    modelA = decisionTree_InfoGain(foldXa,foldya,depth);
    
    yhatA = modelA.predictFunc(modelA,foldXb);
    errorTestA = sum(yhatA ~= foldyb)/foldN
    
    % Train with 2501 - 5000 and test with 1 - 2500
    modelB = decisionTree_InfoGain(foldXb,foldyb,depth);
    
    yhatB = modelB.predictFunc(modelB,foldXa);
    errorTestB = sum(yhatB ~= foldya)/foldN
    
    fprintf('Depth: %d\n',depth);

    resultTrain(depth,2) = errorTestA;

    resultTrain(depth,4) = errorTestB;
    total = errorTestA + errorTestB;
    
    resultTrain(depth,5) = total/maxFold;
end