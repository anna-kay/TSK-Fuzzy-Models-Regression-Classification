format compact
close all
clear
clc

rng(1, 'v4');

%% Evaluation functions
Rsq = @(ypred,y) 1-sum((y-ypred).^2)/sum((y-mean(y)).^2);
Nmse = @(ypred,y) sum((y-ypred).^2)/sum((y-mean(y)).^2);

%% Performance table initialization
Perf=zeros(4,1);

%% Load data - Split data into train set, validation set and test set
data=load('airfoil_self_noise.dat');
preproc=1;
[trainData,valData,testData]=split_scale(data,preproc);

% Split inputs and outputs of each set
X_train = trainData( : , 1:end-1 );
Y_train = trainData( :, end);

X_val = valData( : , 1:end-1 );
Y_val = valData( :, end);

X_test = testData( : , 1:end-1 );
Y_test = testData( :, end);

%% Define fuzzy inference system parameters

options = genfisOptions('GridPartition', ...
    'NumMembershipFunctions', 2, ...
    'InputMembershipFunctionType', 'gbellmf', ...
    'OutputMembershipFunctionType', 'constant');

fis = genfis(X_train, Y_train, options);
 
figure(1);
plotMFs(fis, size(trainData,2)-1);
sgtitle('Initial fuzzy sets');
 
%% Edit fuzzy set overlap

% mfedit(fis)
% fis = readfis('fis');

% figure(2);
% plotMFs(fis, size(trainData,2)-1);

%% Tune fuzzy inference system

epochs = 500;
[trainFis, trainError, ~ , valFis, valError] = anfis(trainData, fis, [epochs 0 0.01 0.9 1.1], [], valData);
 
% Plot errors
figure(2);
plot([trainError valError], 'LineWidth',2); grid on;
xlabel('# of Iterations'); ylabel('Error');
legend('Training Error', 'Validation Error' );
title('Learning Curves');
 
% Final model selection
% "Os teliko montelo epilgetai panta ekeino pou antistoixei sto mikrotero sfalma sto sunolo epikurwsis"
TSK_model_1 = valFis;

% Prediction using the anfis model
Y_test_pred = evalfis(TSK_model_1, X_test);

%% Final fuzzy sets

figure(3);
plotMFs(TSK_model_1, size(trainData,2)-1);
sgtitle('Final fuzzy sets');

%% True-Predicted Values
figure(4);
plot(Y_test_pred, 'o');
hold on;
plot(Y_test);
legend('predicted', 'true');
xlabel('# of testset sample'); ylabel('output value');
title('True-Predicted Values');

%% Prediction error
figure(5);
error = abs(Y_test_pred-Y_test);
bar(error);
xlabel('# of testset sample'); ylabel('absolute error value');
title('Prediction Errors');



%% Metrics computation

RMSE = sqrt(mse(Y_test_pred, Y_test));
NMSE = Nmse(Y_test_pred, Y_test);
NDEI = sqrt(NMSE);
R2 = Rsq(Y_test_pred, Y_test);

%% Results Table

Perf(1:4,:) = [RMSE; NMSE; NDEI; R2];

varnames={'TSK_model_1'};
rownames={'RMSE', 'NMSE', 'NDEI', 'R2'};
Perf=array2table(Perf,'VariableNames',varnames,'RowNames',rownames);
