clc; clear all; close all
% Generate Training data
x = exam4q1_generateData(1000);
% Split in to 10 folds
data_f = zeros(10, size(x,2)/10);
labels_f = zeros(10, size(x,2)/10);
for k = 1:10
    data_f(k, :) = x(1, (100*(k-1)+1):100*k);
    labels_f(k, :) = x(2, (100*(k-1)+1):100*k);
end
% We will test N perceptrons from 1 to 10
validation = zeros(1, 10);
valid_sub = zeros(1, 10);
fold_sub = zeros(1, 9);
for n = 1:10
    % Create the Neural Network model
    layers = [
        sequenceInputLayer(1)
        fullyConnectedLayer(n)
        leakyReluLayer
        fullyConnectedLayer(1)
        regressionLayer
        ];
    options = trainingOptions('adam', 'Verbose', false, 'InitialLearnRate', .01);
    % Train and validate on each for
    for k = 1:10
        train = [data_f(k, :);labels_f(k, :)];
        mdl = trainNetwork(train(1, :), train(2, :), layers, options);
        idx = 1;
        for k_sub = 1:10
            if k ~= k_sub
                pdc = predict(mdl, data_f(k_sub, :));
                fold_sub(1, idx) = mean((labels_f(k_sub, :) - pdc).^2);
                idx = idx + 1;
            end
        end
        valid_sub(1, k) = mean(fold_sub);
    end
    validation(1, n) = mean(valid_sub);
end
figure()
bar(validation)
xlabel('Number of Perceptrons')
ylabel('Mean Square Error')
% Get the minimum
[~, n] = min(validation);
% Train a network with that on the whole set
layers = [
        sequenceInputLayer(1)
        fullyConnectedLayer(n)
        reluLayer
        fullyConnectedLayer(1)
        regressionLayer
        ];
mdl = trainNetwork(x(1, :), x(2, :), layers, options);
% Load a validation set
x_validation = exam4q1_generateData(10000);
output = predict(mdl, x_validation(1, :));
figure()
plot(x_validation(1, :), x_validation(2, :), 'b.')
hold on
plot(x_validation(1, :), output, 'ro')