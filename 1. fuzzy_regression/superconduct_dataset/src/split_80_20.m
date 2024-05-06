%% Split - 80/20

function [trainData, testData] = split_80_20(data)

    dim = size(data);
    dataLength = dim(1);
    
    idx = randperm(dataLength);
    trainIdx=idx(1 : round(length(idx)*0.8) );
    testIdx=idx(round(length(idx)*0.8)+1 : end);
    
    trainData = data(trainIdx, : );
    testData = data(testIdx, :);
    
end