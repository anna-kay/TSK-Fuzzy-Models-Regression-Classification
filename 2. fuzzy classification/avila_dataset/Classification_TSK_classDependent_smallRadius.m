format compact

close all
clear
clc

rng(2019, 'v4');

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

X_trnData = trnData(:, 1:end-1);
Y_trnData = trnData(:, end);

%% Class-Dependent Scatter Partition - Radius = 0.5

radius=0.5;

[c1,sig1]=subclust(trnData(trnData(:,end)==1,:),radius);
[c2,sig2]=subclust(trnData(trnData(:,end)==2,:),radius);
[c3,sig3]=subclust(trnData(trnData(:,end)==3,:),radius);
[c4,sig4]=subclust(trnData(trnData(:,end)==4,:),radius);
[c5,sig5]=subclust(trnData(trnData(:,end)==5,:),radius);
[c6,sig6]=subclust(trnData(trnData(:,end)==6,:),radius);
[c7,sig7]=subclust(trnData(trnData(:,end)==7,:),radius);
[c8,sig8]=subclust(trnData(trnData(:,end)==8,:),radius);
[c9,sig9]=subclust(trnData(trnData(:,end)==9,:),radius);
[c10,sig10]=subclust(trnData(trnData(:,end)==10,:),radius);
[c11,sig11]=subclust(trnData(trnData(:,end)==11,:),radius);
[c12,sig12]=subclust(trnData(trnData(:,end)==12,:),radius);

num_rules=size(c1,1)+size(c2,1)+size(c3,1)+size(c4,1)+size(c5,1)+size(c6,1)+...
    size(c7,1)+ size(c8,1)+size(c9,1)+ size(c10,1)+size(c11,1)+size(c12,1);

%% Model intialization

%Build FIS From Scratch
fis=newfis('FIS_SC','sugeno');

%Add Input-Output Variables
names_in={'in1','in2','in3','in4','in5','in6','in7', 'in8', 'in9','in10'};
for i=1:size(trnData,2)-1
    fis=addvar(fis,'input',names_in{i},[0 1]);
end
fis=addvar(fis,'output','out1',[0 1]);

%Add Input Membership Functions
name='sth';
for i=1:size(trnData,2)-1
    for j=1:size(c1,1)
        fis=addmf(fis,'input',i,name,'gaussmf',[sig1(j) c1(j,i)]);
    end
    for j=1:size(c2,1)
        fis=addmf(fis,'input',i,name,'gaussmf',[sig2(j) c2(j,i)]);
    end
    for j=1:size(c3,1)
        fis=addmf(fis,'input',i,name,'gaussmf',[sig3(j) c3(j,i)]);
    end
    for j=1:size(c4,1)
         fis=addmf(fis,'input',i,name,'gaussmf',[sig4(j) c4(j,i)]);
    end 
    for j=1:size(c5,1)
        fis=addmf(fis,'input',i,name,'gaussmf',[sig5(j) c5(j,i)]);
    end
    for j=1:size(c6,1)
        fis=addmf(fis,'input',i,name,'gaussmf',[sig6(j) c6(j,i)]);
    end
    for j=1:size(c7,1)
        fis=addmf(fis,'input',i,name,'gaussmf',[sig7(j) c7(j,i)]);
    end
    for j=1:size(c8,1)
        fis=addmf(fis,'input',i,name,'gaussmf',[sig8(j) c8(j,i)]);
    end
    for j=1:size(c9,1)
        fis=addmf(fis,'input',i,name,'gaussmf',[sig9(j) c9(j,i)]);
    end
    for j=1:size(c10,1)
        fis=addmf(fis,'input',i,name,'gaussmf',[sig10(j) c10(j,i)]);
    end
    for j=1:size(c11,1)
        fis=addmf(fis,'input',i,name,'gaussmf',[sig11(j) c11(j,i)]);
    end
    for j=1:size(c12,1)
        fis=addmf(fis,'input',i,name,'gaussmf',[sig12(j) c12(j,i)]);
    end    
end


%Add Output Membership Functions

twos = 2.* ones(size(c2,1),1);
threes = 3.* ones(size(c3,1),1);
fours = 4.* ones(size(c4,1),1);
fives = 5.* ones(size(c5,1),1);
sixes = 6.* ones(size(c6,1),1);
sevenths = 7.* ones(size(c7,1),1);
eights = 8.* ones(size(c8,1),1);
nines = 9.* ones(size(c9,1),1);
tens = 10.* ones(size(c10,1),1);
elevens = 11.* ones(size(c11,1),1);
twelves = 12.* ones(size(c12,1),1);

params=[ones(1,size(c1,1)) twos' threes' fours' fives' sixes' sevenths' eights' nines' tens' elevens' twelves'];
for i=1:num_rules
    fis=addmf(fis,'output',1,name,'constant',params(i));
end

%Add FIS Rule Base
ruleList=zeros(num_rules,size(trnData,2));
for i=1:size(ruleList,1)
    ruleList(i,:)=i;
end
ruleList=[ruleList ones(num_rules,2)];
fis=addrule(fis,ruleList);


figure(1)
plotMFs(fis, size(trnData,2)-1);

%Train & Evaluate ANFIS
epochs = 500 ;
[trnFis,trnError,~,valFis,valError] = anfis(trnData,fis,[epochs 0 0.001 0.9 1.1],[],valData);

figure(2)
plot([trnError valError],'LineWidth',2); grid on;
legend('Training Error','Validation Error');
xlabel('# of Epochs');
ylabel('Error');

figure(3)
plotMFs(valFis,size(trnData,2)-1);

Y=evalfis(testData(:,1:end-1),valFis);
Y=round(Y);
diff=testData(:,end)-Y;
Acc=(length(diff)-nnz(diff))/length(Y)*100;

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

save(mfilename, 'C');

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

fprintf('OA = %0.4f \n', OA);
fprintf('k = %0.4f \n', k);

for i=1:length(PA)
  fprintf('PA(%d) = %0.4f \n', i, PA(i));  
end

for i=1:length(UA)
  fprintf('UA(%d) = %0.4f \n', i, UA(i));  
end

save(mfilename, 'UA', 'PA', 'C', 'OA', 'k');
