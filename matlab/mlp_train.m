%% mlp_train.m
% MLP Training and PCS Evaluation for A/B Test Factor Optimization
% Based on Farrell et al. (2021) architecture recommendations
%
% Usage:
%   [pcs_values, sample_sizes] = mlp_train(n_factors, depth, width, n_reps)
%
% Parameters:
%   n_factors  - Number of binary A/B test factors
%   depth      - L: number of hidden layers
%   width      - H: neurons per hidden layer
%   n_reps     - Number of repetitions for PCS estimation

function [pcs_values, sample_sizes] = mlp_train(n_factors, depth, width, n_reps)

    if nargin < 1, n_factors = 5;   end
    if nargin < 2, depth     = 2;   end
    if nargin < 3, width     = 100; end
    if nargin < 4, n_reps    = 100; end

    % ── Ground truth ──────────────────────────────────────────────────────
    rng(0);
    beta      = rand(n_factors, 1) * 2 - 1;   % coefficients in [-1, 1]
    noise_std = 0.5;

    % All 2^n factor combinations
    all_combos = generate_combinations(n_factors);  % (2^n x n)
    true_resp  = all_combos * beta;
    [~, best_idx] = max(true_resp);

    % ── Sample sizes to evaluate ──────────────────────────────────────────
    sample_sizes = [50, 100, 200, 400, 600, 800, 1000, 1500, 2000];
    pcs_values   = zeros(size(sample_sizes));

    fprintf('MLP (L=%d, H=%d, reps=%d)\n', depth, width, n_reps);
    fprintf('%-12s %-8s\n', 'Sample Size', 'PCS');
    fprintf('%s\n', repmat('-', 1, 22));

    for si = 1:length(sample_sizes)
        n = sample_sizes(si);
        correct = 0;

        for rep = 1:n_reps
            rng(42 + rep);
            % Sample binary factor combinations + noisy responses
            X = double(rand(n, n_factors) > 0.5);
            y = X * beta + randn(n, 1) * noise_std;

            % Train MLP and get prediction
            predicted_best = train_and_predict(X, y, depth, width, all_combos);
            if predicted_best == best_idx
                correct = correct + 1;
            end
        end

        pcs_values(si) = correct / n_reps;
        fprintf('%-12d %-8.3f\n', n, pcs_values(si));
    end
end


%% ── Train one MLP and return the predicted best combination index ────────
function best_idx = train_and_predict(X, y, depth, width, all_combos)
    n_factors = size(X, 2);
    lr        = 0.01;
    epochs    = 300;

    % Build layer dimensions
    dims = [n_factors, repmat(width, 1, depth), 1];
    n_layers = length(dims) - 1;

    % He initialisation
    W = cell(n_layers, 1);
    b = cell(n_layers, 1);
    for l = 1:n_layers
        fan_in = dims(l);
        W{l}   = randn(dims(l), dims(l+1)) * sqrt(2 / fan_in);
        b{l}   = zeros(1, dims(l+1));
    end

    % SGD training
    for epoch = 1:epochs
        % Forward pass
        A = cell(n_layers + 1, 1);
        Z = cell(n_layers, 1);
        A{1} = X;
        for l = 1:n_layers
            Z{l} = A{l} * W{l} + b{l};
            if l < n_layers
                A{l+1} = max(0, Z{l});    % ReLU
            else
                A{l+1} = Z{l};             % linear output
            end
        end

        % MSE loss
        y_pred = A{end};
        % Backward pass
        delta = 2 * (y_pred - y) / size(X, 1);
        for l = n_layers:-1:1
            dW = A{l}' * delta;
            db = sum(delta, 1);
            if l > 1
                delta = (delta * W{l}') .* double(Z{l-1} > 0);
            end
            W{l} = W{l} - lr * dW;
            b{l} = b{l} - lr * db;
        end
    end

    % Predict over all combinations
    A_pred = all_combos;
    for l = 1:n_layers
        Z_pred = A_pred * W{l} + b{l};
        if l < n_layers
            A_pred = max(0, Z_pred);
        else
            A_pred = Z_pred;
        end
    end
    [~, best_idx] = max(A_pred);
end


%% ── Generate all 2^n binary combinations ────────────────────────────────
function combos = generate_combinations(n)
    n_rows = 2^n;
    combos = zeros(n_rows, n);
    for i = 1:n_rows
        bits = dec2bin(i-1, n) - '0';
        combos(i, :) = bits;
    end
end
