
% Exam 4 Question 3
clc; clear all; close all;
% Load images
plane_raw = imread('3096_color.jpg');
bird_raw = imread('42049_color.jpg');
% Reshape to an array. Pull using a raster scan so we can get array 
plane_r = reshapeQ3(plane_raw);
bird_r = reshapeQ3(bird_raw);
% Fit Gaussians
plane_gmm = fitgmdist(plane_r, 2);
bird_gmm = fitgmdist(bird_r, 2);
% Run clustering 
plane_clu = cluster(plane_gmm, plane_r);
bird_clu = cluster(bird_gmm, bird_r);
% Reshape
figure()
subplot(2, 2, 1)
imshow(bird_raw)
subplot(2, 2, 3)
imshow(reshape(bird_clu./2, size(bird_raw, 2), size(bird_raw, 1))')
subplot(2, 2, 2)
imshow(plane_raw)
subplot(2, 2, 4)
imshow(reshape(plane_clu./2, size(plane_raw, 2), size(plane_raw, 1))')

%% PART 2 10-Fold to find optimal number of clusters
% Bird
ll = zeros(1, 10);
img = bird_r;
[ax1, ~] = size(img);
fold = floor(ax1/10);
for cla = 1:10
    sub = zeros(1, 10);
    for k = 1:10
        % Subset of training data
        train_sub = img((fold*(k-1)+1):fold*k, :);
        % Subset of validation data
        idx = ismember(img(:, 1), train_sub(:, 1));
        valid_sub = img(~idx, :);
        % Train Model
        mdl = fitgmdist(train_sub, cla, 'Regularize', 0.01, 'Options', statset('MaxIter', 300));
        % Predictions
        pdc = log(pdf(mdl, valid_sub));
        % Accuracy
        sub(k) = mean(-pdc);
    end
    ll(cla) = mean(sub);
end
figure()
bar(ll)
xlabel('Clusters')
ylabel('-Log Likelihood')

% Train final model with best fit
[~, idx] = min(ll);
mdl = fitgmdist(img, idx, 'Regularize', 0.01, 'Options', statset('MaxIter', 300));
% Predictions
bird_clu = cluster(mdl, img);

% Plane
ll = zeros(1, 10);
img = plane_r;
[ax1, ~] = size(img);
fold = floor(ax1/10);
for cla = 1:10
    sub = zeros(1, 10);
    for k = 1:10
        % Subset of training data
        train_sub = img((fold*(k-1)+1):fold*k, :);
        % Subset of validation data
        idx = ismember(img(:, 1), train_sub(:, 1));
        valid_sub = img(~idx, :);
        % Train Model
        mdl = fitgmdist(train_sub, cla, 'Regularize', 0.01, 'Options', statset('MaxIter', 300));
        % Predictions
        pdc = log(pdf(mdl, valid_sub));
        % Accuracy
        sub(k) = mean(-pdc);
    end
    ll(cla) = mean(sub);
end
figure()
bar(ll)
xlabel('Clusters')
ylabel('-Log Likelihood')

% Train final model with best fit
[~, idx] = min(ll);
mdl = fitgmdist(img, idx, 'Regularize', 0.01, 'Options', statset('MaxIter', 300));
% Predictions
plane_clu = cluster(mdl, img);

figure()
subplot(2, 2, 1)
imshow(bird_raw)
subplot(2, 2, 3)
imagesc(reshape(bird_clu./2, size(bird_raw, 2), size(bird_raw, 1))')
subplot(2, 2, 2)
imshow(plane_raw)
subplot(2, 2, 4)
imagesc(reshape(plane_clu./2, size(plane_raw, 2), size(plane_raw, 1))')



function image0 = reshapeQ3(image)
% Get dimensions
[ax1, ax2, ~] = size(image);
% Allocate space
image0 = zeros(ax1*ax2, 5);
for i = 1:ax1
    for j =1:ax2
        row = (i-1)*ax2 + j;
        image0(row, :) = [image(i, j, 1) image(i, j, 2) image(i, j, 3) i j];
    end
end
% Normalize
maxes = max(image0, [], 1);
image0 = image0./maxes;
end