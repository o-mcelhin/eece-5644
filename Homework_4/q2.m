% Take-home 4 Q2
clc; clear all; close all
% Generate training data
[data, label] = generateMultiringDataset(4, 11000);
% Pull training and validation data
data_t = data(:, 1:1000); label_t = label(1:1000); data_v = data(:, 1001:end);
label_v = label(1001:end);
% Use 10-Fold validation to pick sigma
sigma_test = 1:20;
sig_v = zeros(1, 20);
for sig = 1:20
    sub = zeros(1, 10);
    for k = 1:10
        % Subset of training data
        train_sub = data_t(:, (100*(k-1)+1):100*k);
        % Subset of validation data
        idx = ismember(data_t(1,:), train_sub(1,:));
        valid_sub = data_t(:, ~idx);
        % Train Model
        t = templateSVM('KernelFunction', 'gaussian', 'KernelScale', sigma_test(sig), 'BoxConstraint', 5);
        % Train
        [mdl, opt] = fitcecoc(train_sub', label_t(idx)', 'Learners', t);
        % Predictions
        pdc = predict(mdl, valid_sub');
        % Accuracy
        acc = sum(pdc' == label_t(~idx))/size(pdc, 1);
        sub(k) = acc;
    end
    sig_v(sig) = mean(sub);
end
figure()
bar(sigma_test, sig_v)
xlabel('Sigma')
ylabel('P(Hit)')
% Use 10-Fold validation to pick C
C_test = 1:10;
c_v = zeros(1, 10);
for sig = 1:10
    sub = zeros(1, 10);
    for k = 1:10
        % Subset of training data
        train_sub = data_t(:, (100*(k-1)+1):100*k);
        % Subset of validation data
        idx = ismember(data_t(1,:), train_sub(1,:));
        valid_sub = data_t(:, ~idx);
        % Train Model
        t = templateSVM('KernelFunction', 'gaussian', 'KernelScale', 10, 'BoxConstraint', C_test(sig));
        % Train
        [mdl, opt] = fitcecoc(train_sub', label_t(idx)', 'Learners', t);
        % Predictions
        pdc = predict(mdl, valid_sub');
        % Accuracy
        acc = sum(pdc' == label_t(~idx))/size(pdc, 1);
        sub(k) = acc;
    end
    c_v(sig) = mean(sub);
end
figure()
bar(C_test, c_v)
xlabel('Box Parameter C')
ylabel('P(Hit)')
% Load in a Gaussian Kernel
t = templateSVM('KernelFunction', 'gaussian', 'KernelScale', 20, 'BoxConstraint', 5);
% Train
[mdl, opt] = fitcecoc(data_t', label_t', 'Learners', t);

% Predictions
pdc = predict(mdl, data_v');
acc = sum(pdc == label_v')/size(label_v,2)

% Plot error
figure()

idx = ((pdc' == 1) & (label_v == 1));
plot(data_v(1, idx), data_v(2, idx), '.')
hold on

idx = ((pdc' == 2) & (label_v == 2));
plot(data_v(1, idx), data_v(2, idx), '.')

idx = ((pdc' == 3) & (label_v == 3));
plot(data_v(1, idx), data_v(2, idx), '.')

idx = ((pdc' == 4) & (label_v == 4));
plot(data_v(1, idx), data_v(2, idx), '.')

idx = (pdc' ~= label_v);
plot(data_v(1, idx), data_v(2, idx), 'r.')
