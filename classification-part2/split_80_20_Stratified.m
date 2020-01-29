%% Split - 80/20

function [trainData, testData] = split_80_20_Stratified(data)

    group1 = data(:, end);
    
    C1 = cvpartition(group1, 'HoldOut', 0.2);
      
    trainIdx=C1.training;
    testIdx=C1.test;
    
    trainData = data(trainIdx, : );
    testData = data(testIdx, :);
    
end