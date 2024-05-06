format compact
close all
clear
clc

rng(2005, 'v4');

%% Load data - Normalization -Split data
data=load('avila.txt');

% Normalization
X_data = data(:, 1:end-1);
xmin=min(X_data,[],1);
xmax=max(X_data,[],1);
X_data=(X_data-repmat(xmin,[length(X_data) 1]))./(repmat(xmax,[length(X_data) 1])-repmat(xmin,[length(X_data) 1]));

data = [X_data, data(:, end)];

% partition of the whole dataset into 2 stratified sets 80/20
group1 = data(:, end);

C1 = cvpartition(group1, 'HoldOut', 0.2);
trainIdx = C1.training;
testIdx = C1.test;

trainData = data(trainIdx, :);
testData = data(testIdx, :);

%  partition of the test set into 2 stratified sets 80/20

group2 = trainData(:, end);

C2 = cvpartition(group2, 'HoldOut', 0.2);
trainIdx = C2.training;
testIdx = C2.test;

trnData = trainData(trainIdx, :);
valData = trainData(testIdx, :);

%% Class-Independent Scatter Partition - Radius = 0.8

radius = 0.8;

X_trnData = trnData(:, 1:end-1);
Y_trnData = trnData(:, end);

% model definition & initialization

options = genfisOptions('SubtractiveClustering', 'ClusterInfluenceRange', radius);                                 
fis = genfis(X_trnData, Y_trnData, options);
mfedit(fis)

%% After changing output function to singleton ('constant')

fis = readfis('sug101');

% Plot inputs before training
figure(1);
plotMFs(fis, size(trnData,2)-1);
sgtitle('Initial Fuzzy Sets');

%  model training
[trainFis, trainError, ~, valFis, valError] = anfis(trnData, fis, [500 0 0.001 0.9 1.1], [], valData);

%% Plot inputs after training
figure(2);
plotMFs(valFis,size(trnData,2)-1);
sgtitle('Final Fuzzy Sets');

% Plot learning curves
figure(3);
plot([trainError valError],'LineWidth',2); grid on;
xlabel('# of Iterations'); ylabel('Error');
legend('Training Error','Validation Error');
title('ANFIS Hybrid Training - Validation');

%%  Evaluation - Metrics

Y_testData = testData(:,end);
Y_pred=evalfis(testData(:,1:end-1),valFis);
Y_pred=round(Y_pred);

% bounding the predicted values between 1 and 12
for i=1:length(Y_pred)
    if (Y_pred(i)<min(Y_testData))
        Y_pred(i)=min(Y_testData);
    elseif (Y_pred(i)>max(Y_testData))
            Y_pred(i)=max(Y_testData);
    end
end

diff=Y_testData-Y_pred;
Acc=(length(diff)-nnz(diff))/length(Y_pred)*100;

% Error matrix
C = confusionmat(Y_testData, Y_pred);
% confusionchart(C);

numOfClasses = 12;
numOfSamples = length(Y_testData);

trueC = repmat(zeros, [numOfClasses 1]);
predictedC = repmat(zeros, [numOfClasses 1]);

sumOfDiag = trace(C);
prodTruePred = repmat(zeros, [numOfClasses 1]);

PA = repmat(zeros, [numOfClasses 1]);
UA = repmat(zeros, [numOfClasses 1]);

OA = sumOfDiag/numOfSamples;

for i= 1:numOfClasses
    trueC(i) = sum(C(i,:));
    predictedC(i) = sum(C(:,i));
end

for i = 1: numOfClasses
    PA(i) = C(i,i)/trueC(i); 
end
    
for i = 1: numOfClasses
    UA(i) = C(i,i)/predictedC(i); 
end

for i = 1: numOfClasses
   prodTruePred(i) = trueC(i)*predictedC(i);
end

k = (numOfSamples*sumOfDiag - sum(prodTruePred))/(numOfSamples^2  -sum(prodTruePred));

figure(4);
confusionchart(C);
title('Error Matrix');

fprintf('OA = %0.4f \n', OA);
fprintf('k = %0.4f \n', k);

for i=1:length(PA)
  fprintf('PA(%d) = %0.4f \n', i, PA(i));  
end

for i=1:length(UA)
  fprintf('UA(%d) = %0.4f \n', i, UA(i));  
end