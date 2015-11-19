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

%% QUESTION 2 - JULI�
%D1
data1 = struct('x',replaceNaNbyMean(x),'y',y);
%D2
data2 = struct('x',replaceNaNbyMeanOfClass(x,y),'y',y);

%Means
means1 = mean(data1.x,2);
means2 = mean(data2.x,2);

%% QUESTION 3 - XAVI
x_v2 = [ones(1,num_instances);x(7:8,:)];
w = analyticLinearRegression(x_v2,y);

% Plane normal vector
w(2:3)

% Threshold
w(1)

%% QUESTION 4 - JULI�
%a)
clear all;
close all;
clc;
data = load('../Data/diabetes');
x = data.x;
y = data.y;
%b)
data2 = struct('x',replaceNaNbyMeanOfClass(x,y),'y',y);
%c)
sizeTrain = ceil(4 * size(data.x,2)/5);
sizeTest = size(data.x,2)-sizeTrain;

data2train = struct('x',data2.x(:,1:sizeTrain),'y',data2.y(:,1:sizeTrain));

%d

%e

%% QUESTION 5 - JULI�


%% QUESTION 6 - XAVI