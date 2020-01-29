format compact
clear
close all
clc

rng(2019, 'v4');

%% Load initial dataset

data = load('superconduct.csv');

%% Normalization of Inputs of data

% Separating the inputs and outputs
X_data = data(:, 1:end-1);
Y_data = data(:, end);

% Normalization to unit hypercube
xmin=min(X_data,[],1);
xmax=max(X_data,[],1);

X_data=(X_data-repmat(xmin,[length(X_data) 1]))./(repmat(xmax,[length(X_data) 1])-repmat(xmin,[length(X_data) 1]));

data = [X_data Y_data];



%% Feature Selection (via Relief)
% relieff algorithm is applied only on the train data (to avoid learning the test data)
% output y is numeric in our case so so relieff performs RReliefF analysis
% for regression by default
% 10 is the number of nearest neighbours

% [ranksReg, weights] = relieff(X_data, Y_data, 30);

% ranks -> stores the indices of the selected features in descending
% order of importance

% save(mfilename, 'ranksReg');

load('ranks.mat');

%% Select portion of the dataset
my_data = data;


%% Split intial dataset into trainData->80% and testData->20%
[trainData, testData] = split_80_20(my_data);

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
errorGoal = 0;
stepSize = 0.01;
decreaseRate = 0.9;
increaseRate = 1.1;

G1 = 3:10; % G1 = 5; 
G2 = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]; % G2 = 0.3;
gridSearchData = zeros(length(G1)*length(G2), 3); 

%% Grid Search Section
numberOfRound = 1;

for numOfFeatures = G1
   mostRelevantFeatures = ranksReg(1:numOfFeatures);
   disp(mostRelevantFeatures);
   X_trainDataSelectedFeatures = X_trainData(:, mostRelevantFeatures);
   
   % combine inputs & outputs
   trainDataSelectedFeatures = [X_trainDataSelectedFeatures, Y_trainData];
   
   for radius = G2
         %% Grid Search Section -> Cross-validation Subsection
           
           % to store the minimum validation for the current Influence
           % Range, Ra
           minValError = [0 0 0 0 0]; 
         
           % 5-fold cross-validation loop 
           for i=1:5 
               % split again our train data, into train and validation sets
               [trnData, valData] = split_80_20(trainDataSelectedFeatures);
               % separate inputs and outputs
               X_trnData = trnData(:, 1:end-1);
               Y_trnData = trnData(:, end);

               X_valData = valData(:, 1:end-1);
               Y_valData = valData(:, end);

               % model definition
               options = genfisOptions('SubtractiveClustering', 'ClusterInfluenceRange', radius);
               fis = genfis(X_trnData, Y_trnData, options);
               [trainFis, trainError, ~, valFis, valError] = anfis(trnData, fis, [epochs errorGoal stepSize decreaseRate increaseRate], [], valData);
               
               % valError stores error for all epochs of training, we take
               % into only the smallest one
             
               % store minimum validation error for each iteration of 
               % the 5-fold cross validation
               minValError(i) = min(valError);
               numOfRules = max(size(fis.rule));
           end
         %% end of cross-validation section
           
         gridSearchData(numberOfRound, 1:4) = [numOfFeatures, radius, mean(minValError), numOfRules];
         numberOfRound = numberOfRound+1;
   end
   
end
 
save('gridSearchData.mat', 'gridSearchData');
varnames = {' NumOfFeatures', 'InfluenceRange', 'MinValError', 'NumOfRules'};
Results = array2table(gridSearchData, 'VariableNames', varnames);

writetable(Results, 'resultsTable.xls');
 
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
mostRelevantFeatures = ranksReg(1:bestNumOfFeatures);
 X_trainDataSelectedFeatures = X_trainData(:, mostRelevantFeatures);
 X_testDataSelectedFeatures = X_testData(:, mostRelevantFeatures);
 
% combine inputs & outputs
trainDataSelectedFeatures = [X_trainDataSelectedFeatures, Y_trainData];
testDataSelectedFeatures = [X_testDataSelectedFeatures, Y_testData];

[trnData, valData] = split_80_20(trainDataSelectedFeatures);
 % separate inputs and outputs
 X_trnData = trnData(:, 1:end-1);
 Y_trnData = trnData(:, end);

 X_valData = valData(:, 1:end-1);
 Y_valData = valData(:, end);

% the training
options = genfisOptions('SubtractiveClustering', 'ClusterInfluenceRange', bestRadius);
finalFis = genfis(X_trnData, Y_trnData, options);

% plots initial fuzzy sets
figure(1);
plotMFs(finalFis, size(X_trainDataSelectedFeatures,2));
title('Initial fuzzy sets');

[trainFis, trainError, ~, valFis, valError] = anfis(trnData, finalFis,  [epochs errorGoal stepSize decreaseRate increaseRate], [], valData);

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

%% Plot errors

%% True-Predicted Values
figure(4);
plot(Y_test_pred, 'o');
hold on;
plot(Y_testData);
legend('predicted', 'true');
xlabel('# of testset sample'); ylabel('output value');
title('True-Predicted Values');
xlim([0, size(Y_testData, 1)]);
 
%% Prediction error
figure(5);
error = abs(Y_test_pred-Y_testData);
bar(error);
xlabel('# of testset sample'); ylabel('absolute error value');
title('Prediction Errors');

figure(6);
plot3(gridSearchData(:, 1), gridSearchData(:, 2), gridSearchData(:, 3), 'x');
grid on;
xlabel('# of features');
ylabel('Radius');
zlabel('Error')
title('Error as a result of the number of features & radius');

%% Metrics computation

Rsq = @(ypred,y) 1-sum((ypred-y).^2)/sum((y-mean(y)).^2);
Nmse = @(ypred,y) sum((ypred-y).^2)/sum((y-mean(y)).^2);

RMSE = sqrt(mse(Y_test_pred, Y_testData));
NMSE = Nmse(Y_test_pred, Y_testData);
NDEI = sqrt(NMSE);
R2 = Rsq(Y_test_pred, Y_testData);

save(mfilename, 'finalFis', 'RMSE', 'NMSE', 'NDEI', 'R2', 'Results', 'minErrorTriplet',  'gridSearchData');
