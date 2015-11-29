%% Reset all
clear all;
close all;

%% Move to working directory
tmp = matlab.desktop.editor.getActive;
cd(fileparts(tmp.Filename));

%% Load Data
data = load('../Data/diabetes');
x = data.x;
y = data.y;

%% QUESTION 1
num_instances = size(x,2);
dimensionality = size(x,1);
means = mean(x,2);

%% QUESTION 2 - JULIÀ
%D1
D1 = struct('x',replaceNaNbyMean(x),'y',y);
%D2
D2 = struct('x',replaceNaNbyMeanOfClass(x,y),'y',y);

%Means
means1 = mean(D1.x,2);
means2 = mean(D2.x,2);

%% QUESTION 3 - XAVI
x_v2 = [ones(1,num_instances); D1.x];
x_v3 = [ones(1,num_instances); D2.x];
w = analyticLinearRegression(x_v2,y);
w_2 = analyticLinearRegression(x_v3,y);

% Plane normal vector
w(2:9)

% Threshold
w(1)

pred_y_train = double((x_v2'*w)>0);
pred_y_train(find(pred_y_train==0))=-1;

pred_y_train_2 = double((x_v3'*w_2)>0);
pred_y_train_2(find(pred_y_train_2==0))=-1;

differences_train = find(pred_y_train~=D2.y);

% Error rate
confMatTrain = confusionMatrix(pred_y_train,D1.y);
confMatTrain_2 = confusionMatrix(pred_y_train_2,D2.y);
errRateTrain = (confMatTrain(1,2)+confMatTrain(2,1))/size(x_v2,2);
errRateTrain_2 = (confMatTrain_2(1,2)+confMatTrain_2(2,1))/size(x_v2,2);


%% QUESTION 4 - JULIÀ
%a)
clear all;
close all;
clc;
data = load('../Data/diabetes');
x = data.x;
y = data.y;
%b)
D2 = struct('x',replaceNaNbyMeanOfClass(x,y),'y',y);
%c)
sizeTrain = ceil(4 * size(data.x,2)/5);
sizeTest = size(data.x,2)-sizeTrain;

D2train = struct('x',D2.x(:,1:sizeTrain),'y',D2.y(1:sizeTrain,:));
D2test = struct('x',D2.x(:,sizeTrain+1:end),'y',D2.y(sizeTrain+1:end,:));

%d)
x_v3 = [ones(1,sizeTrain);D2train.x];
x_v4 = [ones(1,sizeTest);D2test.x];
w = analyticLinearRegression(x_v3,(D2train.y));

%e)
% TRAIN
pred_y_train = double((x_v3'*w)>0);
pred_y_train(find(pred_y_train==0))=-1;

differences_train = find(pred_y_train~=D2train.y);

% Error rate
confMatTrain = confusionMatrix(pred_y_train,D2train.y);
errRateTrain = (confMatTrain(1,2)+confMatTrain(2,1))/sizeTrain;


% TEST
pred_y_test = double((x_v4'*w)>0);
pred_y_test(find(pred_y_test==0))=-1;

differences_test = find(pred_y_test~=D2test.y);

% Error rate
confMatTest = confusionMatrix(pred_y_test,D2test.y);
errRateTest = (confMatTest(1,2)+confMatTest(2,1))/sizeTest;

%% QUESTION 5 - JULIÀ
%a)
clear all;
close all;
clc;
data = load('../Data/diabetes');
x = data.x;
y = data.y;

%b)
sizeTrain = ceil(4 * size(data.x,2)/5);
sizeTest = size(data.x,2)-sizeTrain;
D2train = struct('x',data.x(:,1:sizeTrain),'y',data.y(1:sizeTrain,:));
D2test = struct('x',data.x(:,sizeTrain+1:end),'y',data.y(sizeTrain+1:end,:));

%c)
D2train = ...
    struct('x',replaceNaNbyMeanOfClass(D2train.x,D2train.y),'y',D2train.y);
D2test = ...
    struct('x',replaceNaNbyMeanOfClassTrain(D2test.x, D2test.y, ...
    D2train.x, D2train.y),'y',D2test.y);

%d)
x_v3 = [ones(1,sizeTrain);D2train.x];
x_v4 = [ones(1,sizeTest);D2test.x];
w = analyticLinearRegression(x_v3,(D2train.y));

%e)
% Train
pred_y_train = double((x_v3'*w)>0);
pred_y_train(find(pred_y_train==0))=-1;

differences_train = find(pred_y_train~=D2train.y);
    % Error rate
confMatTrain = confusionMatrix(pred_y_train,D2train.y);
errRateTrain = (confMatTrain(1,2)+confMatTrain(2,1))/sizeTrain;
% Test
pred_y_test = double((x_v4'*w)>0);
pred_y_test(find(pred_y_test==0))=-1;

differences_test = find(pred_y_test~=D2test.y);
    % Error rate
confMatTest = confusionMatrix(pred_y_test,D2test.y);
errRateTest = (confMatTest(1,2)+confMatTest(2,1))/sizeTest;

%% QUESTION 6
clear all;
close all;
clc;
data = load('../Data/diabetes');
x = data.x;
y = data.y;

percentages_train = [0.25 0.4 0.5 0.75 0.90 1];
num_instances = size(data.x,2);

h1 = figure('name','Evolution of errors');
ylim([0 0.5]);
xlim([0 1.2]);
hold on;

h2 = figure('name','Evolution of errors with bound');
ylim([0 0.5]);
xlim([0 1.2]);
hold on;

total_error_trains = zeros(1,size(percentages_train,2));
total_error_tests = zeros(1,size(percentages_train,2));
error_bounds = zeros(1,size(percentages_train,2));
for i = 1:size(percentages_train,2)
    percentage_train = percentages_train(i);
    sizeTrain = ceil(percentage_train * num_instances);
    sizeTest = size(data.x,2)-sizeTrain;
    
    D2train = struct('x',data.x(:,1:sizeTrain),'y',data.y(1:sizeTrain));
    D2test = struct('x',data.x(:,sizeTrain+1:end),'y',data.y(sizeTrain+1:end));

    D2train.x = replaceNaNbyMeanOfClass(D2train.x,D2train.y);
    
    D2test.x = replaceNaNbyMeanOfClassTrain(D2test.x, D2test.y, ...
        D2train.x, D2train.y);

    x_v3 = [ones(1,sizeTrain);D2train.x];
    x_v4 = [ones(1,sizeTest);D2test.x];
    w = analyticLinearRegression(x_v3,D2train.y);


    % TRAIN
    pred_y_train = double((x_v3'*w)>0);
    pred_y_train(pred_y_train==0)=-1;

    differences_train = find(pred_y_train~=D2train.y);
    % Error rate
    confMatTrain = confusionMatrix(pred_y_train,D2train.y);
    errRateTrain = (confMatTrain(1,2)+confMatTrain(2,1))/sizeTrain;
    total_error_trains(i) = errRateTrain;
    
    figure(h1);
    scatter(percentage_train,errRateTrain,'MarkerEdgeColor',[0.5 .5 0]);
    figure(h2);
    scatter(percentage_train,errRateTrain,'MarkerEdgeColor',[0.5 .5 0]);
    
    % TEST
    pred_y_test = double((x_v4'*w)>0);
    pred_y_test(pred_y_test==0)=-1;

    differences_test = find(pred_y_test~=D2test.y);
    % Error rate
    confMatTest = confusionMatrix(pred_y_test,D2test.y);
    errRateTest = (confMatTest(1,2)+confMatTest(2,1))/sizeTest;
    total_error_tests(i) = errRateTest;
    
    figure(h1);
    scatter(percentage_train,errRateTest,'MarkerEdgeColor',[0 0.5 0.5]);
    figure(h2);
    scatter(percentage_train,errRateTest,'MarkerEdgeColor',[0 0.5 0.5]);
    
    error_bounds(i) = errorBound(errRateTrain,3,sizeTrain,0.05);
end
figure(h1);
legend('Train Error','Test Error');

figure(h2);
plot(percentages_train,error_bounds);
legend('Train Error','Test Error','Error Bound');



num_samples = zeros(1,4);

vc = 3;
confidence = 0.95;
dev_error = 0.01;
num_samples(1) = getExpectedNumSamples(dev_error,vc,confidence);

confidence = 0.5;
dev_error = 0.01;
num_samples(2) = getExpectedNumSamples(dev_error,vc,confidence);

confidence = 0.95;
dev_error = 0.05;
num_samples(3) = getExpectedNumSamples(dev_error,vc,confidence);

confidence = 0.95;
dev_error = 0.1;
num_samples(4) = getExpectedNumSamples(dev_error,vc,confidence);

num_samples