format compact
clear
close all
clc

rng(3000, 'v4');

%% Load initial dataset

% Classification_Grid_Search.mat stores in  'ranksClass30' the results of
% relieff algorithm with 30 nearest neighbours

data = load('isolet.dat');

%% Normalization of Inputs of data

% Separating the inputs and outputs
X_data = data(:, 1:end-1);
Y_data = data(:, end);

% Normalization to unit hypercube
xmin=min(X_data,[],1);
xmax=max(X_data,[],1);

X_data=(X_data-repmat(xmin,[length(X_data) 1]))./(repmat(xmax,[length(X_data) 1])-repmat(xmin,[length(X_data) 1]));

data = [X_data Y_data];

% Feature Selection (via Relief)
% relieff algorithm is applied only on the train data (to avoid learning the test data)
% output y is numeric in our case so so relieff performs RReliefF analysis
% for regression by default
% 10 is the number of nearest neighbours

%[ranksClass, weights] = relieff(X_data, Y_data, 10, 'method','classification');

% ranksClass -> stores the indices of the selected features in descending
% order of importance

% save('ranks.mat', 'ranksClass');

load('ranks.mat');

my_data = data(:, :);

%% Split intial dataset into trainData->80% and testData->20%
[trainData, testData] = split_80_20_Stratified(my_data);

% to train set pou 8a ksanaspasei se trn + val mesa sto cross-validation
X_trainData = trainData(:, 1:end-1);
Y_trainData = trainData(:, end);

% to teliko test set
X_testData = testData(:, 1:end-1);
Y_testData = testData(:, end);

%% Initialization of G1, G2 and initialization of gridSearchData array
% G1: set of values for numOfFeatures variable)
% G2: set of values for Ra variable)
% gridSearchData:  stores all triplets of  numOfFeatures, Ra and mean(min_val_error)
% gridSearchData size = (length(G1)*length(G2), 3)

epochs = 500;
stepSize = 0.001;
G1 = 5:20;
G2 = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]; % G2 = 0.3;
gridSearchData = zeros(length(G1)*length(G2), 3); 

%% Grid Search Section
numberOfRound = 1;

for numOfFeatures = G1
   mostRelevantFeatures = ranksClass(1:numOfFeatures);
   disp(mostRelevantFeatures);
   X_trainDataSelectedFeatures = X_trainData(:, mostRelevantFeatures);
   
   % combine inputs & outputs
   trainDataSelectedFeatures = [X_trainDataSelectedFeatures, Y_trainData];
   
   for Ra = G2
         %% Grid Search Section -> Cross-validation Subsection
           
           % to store the minimum validation for the current Influence
           % Range, Ra
           minValError = [0 0 0 0 0]; 
         
           % 5-fold cross-validation loop 
           for i=1:5 
               % split again our train data, into train and validation sets
               [trnData, valData] = split_80_20_Stratified(trainDataSelectedFeatures);
               % separate inputs and outputs
               X_trnData = trnData(:, 1:end-1);
               Y_trnData = trnData(:, end);

               X_valData = valData(:, 1:end-1);
               Y_valData = valData(:, end);

               % model definition
               options = genfisOptions('SubtractiveClustering', 'ClusterInfluenceRange', Ra);
               fis = genfis(X_trnData, Y_trnData, options);
               [trainFis, trainError, ~, valFis, valError] = anfis(trnData, fis, [epochs 0 stepSize 0.9 1.1], [], valData);
               
               % valError stores error for all epochs of training, we take
               % into only the smallest one
             
               % store minimum validation error for each iteration of 
               % the 5-fold cross validation
               minValError(i) = min(valError);
               numOfRules = max(size(fis.rule));
               
           end
         %% end of cross-validation section
           
         gridSearchData(numberOfRound, 1:4) = [numOfFeatures, Ra, mean(minValError), numOfRules];
         numberOfRound = numberOfRound+1;
   end
   
end
 
save('gridSearchData.mat', 'gridSearchData');
varnames = {' NumOfFeatures', 'InfluenceRange', 'MinValError', 'NumOfRules'};
Results = array2table(gridSearchData, 'VariableNames', varnames);
 
%% Minimal Triplet

minErrors = gridSearchData(:, 3);
actualMinError = min(minErrors);
idx = find(minErrors==actualMinError);

minErrorTriplet = gridSearchData(idx, :);

bestNumOfFeatures= minErrorTriplet(1);
bestRadius = minErrorTriplet(2);

save('minTriplet.mat', 'minErrorTriplet');


%% Final ANIFS training

% select only parts of the datasets that correspond to the selected
% features

mostRelevantFeatures = ranksClass(1:bestNumOfFeatures);
 X_trainDataSelectedFeatures = X_trainData(:, mostRelevantFeatures);
 X_testDataSelectedFeatures = X_testData(:, mostRelevantFeatures);
 
% combine inputs & outputs
trainDataSelectedFeatures = [X_trainDataSelectedFeatures, Y_trainData];
testDataSelectedFeatures = [X_testDataSelectedFeatures, Y_testData];

[trnData, valData] = split_80_20_Stratified(trainDataSelectedFeatures);
 % separate inputs and outputs
 X_trnData = trnData(:, 1:end-1);
 Y_trnData = trnData(:, end);

 X_valData = valData(:, 1:end-1);
 Y_valData = valData(:, end);
% 
% % the training
% options = genfisOptions('SubtractiveClustering', 'ClusterInfluenceRange', bestRadius);
finalFis = genfis(X_trnData, Y_trnData, options);

%% mfedit(finalFis);

%% After changing Linear -> Constant

%% finalFis = readfis('sug191.fis');


% plots initial fuzzy sets
figure(1);
plotMFs(finalFis, size(X_trainDataSelectedFeatures,2));
title('Initial fuzzy sets');

[trainFis, trainError, ~, valFis, valError] = anfis(trnData, finalFis, [epochs 0 stepSize 0.9 1.1], [], valData);

% plots final fuzzy sets
figure(2);
plotMFs(finalFis, size(X_trainDataSelectedFeatures,2));
title('Final fuzzy sets');

% plot learning curves
figure(3);
plot([trainError valError], 'LineWidth',2); grid on;
xlabel('# of Iterations'); ylabel('Error');
legend('Training Error', 'Validation Error' );
title('Learning Curves');

%% FINAL EVALUATION

% % Final model selection
% % "Os teliko montelo epilgetai panta ekeino pou antistoixei sto mikrotero sfalma sto sunolo epikurwsis"
myFinalModel = valFis;
Y_test_pred = evalfis(X_testDataSelectedFeatures, myFinalModel);
Y_pred=round(Y_test_pred);

%% Plot errors
%%  Evaluation - Metrics

% bounding the predicted values between 1 and 26
for i=1:length(Y_pred)
    if (Y_pred(i)<min(Y_testData))
        Y_pred(i)=min(Y_testData);
    elseif (Y_pred(i)>max(Y_testData))
            Y_pred(i)=max(Y_testData);
    end
end

%% True-Predicted Values

figure(4);
plot(Y_pred, 'o');
hold on;
plot(Y_testData, 'x');
xlim([0 size(Y_testData,1)]);
legend('predicted', 'true');
xlabel('# of testset sample'); ylabel('output value');
title('True-Predicted Values');

%% Metrics

diff=Y_testData-Y_pred;
Acc=(length(diff)-nnz(diff))/length(Y_pred)*100;

% Error matrix
C = confusionmat(Y_testData, Y_pred);
% confusionchart(C);

numOfClasses = 26;
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

fprintf('OA = %0.4f \n', OA);
fprintf('k = %0.4f \n', k);

for i=1:length(PA)
  fprintf('PA(%d) = %0.4f \n', i, PA(i));  
end

for i=1:length(UA)
  fprintf('UA(%d) = %0.4f \n', i, UA(i));  
end

save(mfilename, 'finalFis', 'OA', 'PA', 'UA', 'k' , 'C',  'gridSearchData', 'minErrorTriplet');
