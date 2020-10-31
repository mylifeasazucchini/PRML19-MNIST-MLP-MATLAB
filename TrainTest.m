close all;
clear all;
rng('default'); %In order to comapre results for different values of parameters, rng is set to 'default'

load('mat1616.mat');
% X is the data matrix
X = mat1616;
clearvars mat1616

% Y corresponds to the desired outputs
Y = [zeros(1,100) ones(1,100) repmat(2,1,100) repmat(3,1,100) repmat(4,1,100)...
    repmat(5,1,100) repmat(6,1,100) repmat(7,1,100) repmat(8,1,100) repmat(9,1,100)];

%% Plot 6 random samples

% % figure(1)
% % for ii = 1:6
% %     subplot(2,3,ii)
% %     rand_num = randperm(1000,1); %select one random number
% %     imshow(uint8(reshape(X(:,rand_num),16,16))) %we reshape the 256 vector in 16x16 matrix to show it
% %     title((Y(rand_num)),'FontSize',20)
% %     axis off
% % end
% % colormap gray

%% Transforming a bit data matrix X

% All the white value have now the value 1 and black pixels value 0
% Classifiers work better with binary matrix
for i=1:size(X,1)
    for j=1:size(X,2)
        a = X(i,j);
        if a ~= 255
            X(i,j) = 0;
        else
            X(i,j) = 1;
        end
    end
end

clearvars a

%% Principal component analysis to reduce dimensions of the samples

%some pixels are useless to characterize corresponding numbers
%that is why we reduce dimension so we deal with less data
[coeff,score,latent] = pca(X');
latent = latent/sum(latent);
index = find(cumsum(latent)>0.95); % we keep dimensions that represent 95% of the data
index(1) %we keep 132 dimensions out of 256
X=score(:,1:index(1))';

%% Dividing the training and testing sets

n = 1000; %number of samples
P = 0.80 ; % 80% of the data for training 
idx = randperm(n); % random division

Xtrain = X(:,idx(1:round(P*n)));
Ytrain = Y(idx(1:round(P*n)));
Xtest = X(:,idx(round(P*n)+1:end));
Ytest = Y(idx(round(P*n)+1:end));

clearvars n P idx

%% MultiLayer Perceptron

%class 0 becomes class 10 in order to deal with output matrix
Ytrain(Ytrain == 0) = 10;
Ytest(Ytest == 0) = 10;

tic
[ypred, t, wHidden, wOutput] = mlp(Xtrain, Ytrain, Xtest, 8000);
execution_time = toc %execution time of the classifier

accuracy = sum(Ytest == ypred)/length(Ytest)

%Confusion matrix
% C = confusionmat(Ytest,ypred);
% confusionchart(C,[1:9 0]);