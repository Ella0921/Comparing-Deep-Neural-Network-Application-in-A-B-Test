%% mlp_iterative.m
% Iterative Data Addition Experiment — Bug Diagnosis & Fix
%
% Reproduces Figure 3 from the paper AND explains why PCS plateaued at ~0.5.
%
% ROOT CAUSE: original code retrained the model on the *latest batch* only,
%             discarding all previously accumulated observations.
% FIX:        retrain from scratch on the full accumulated dataset each time.
%
% Usage:
%   mlp_iterative()          % runs both buggy and fixed versions
%   mlp_iterative(true)      % fixed version only

function mlp_iterative(fixed_only)
    if nargin < 1, fixed_only = false; end

    n_factors  = 5;
    depth      = 2;
    width      = 100;
    n_reps     = 100;
    max_samples = 2000;
    batch_size  = 100;
    noise_std   = 0.5;

    rng(0);
    beta       = rand(n_factors, 1) * 2 - 1;
    all_combos = generate_combinations(n_factors);
    true_resp  = all_combos * beta;
    [~, best_idx] = max(true_resp);

    sizes = batch_size:batch_size:max_samples;

    if ~fixed_only
        fprintf('Running BUGGY version (train only on last batch)...\n');
        pcs_bug = run_iterative(sizes, n_factors, depth, width, ...
                                beta, best_idx, all_combos, ...
                                n_reps, batch_size, noise_std, false);
    end

    fprintf('Running FIXED version (retrain on full dataset)...\n');
    pcs_fix = run_iterative(sizes, n_factors, depth, width, ...
                            beta, best_idx, all_combos, ...
                            n_reps, batch_size, noise_std, true);

    % ── Plot ─────────────────────────────────────────────────────────────
    figure('Name', 'Iterative Experiment: Bug vs Fix', ...
           'Position', [100, 100, 800, 450]);
    hold on;

    if ~fixed_only
        plot(sizes, pcs_bug, 'o-', 'Color', [0.9 0.2 0.2], ...
             'LineWidth', 2, 'MarkerSize', 5, ...
             'DisplayName', 'Bug: train only on last batch');
    end

    plot(sizes, pcs_fix, 's-', 'Color', [0.26 0.63 0.28], ...
         'LineWidth', 2, 'MarkerSize', 5, ...
         'DisplayName', 'Fix: retrain on full dataset');

    yline(0.5, '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.2, ...
          'DisplayName', 'PCS = 0.5 (chance level)');

    xlabel('Cumulative Sample Size', 'FontSize', 12);
    ylabel('PCS', 'FontSize', 12);
    title('Iterative Data Addition — Bug Diagnosis & Fix', 'FontSize', 13);
    ylim([0 1.05]);
    legend('Location', 'southeast', 'FontSize', 10);
    grid on; grid minor;

    saveas(gcf, fullfile('results', 'matlab_fig_iterative.png'));
    fprintf('Saved → results/matlab_fig_iterative.png\n');
end


%% ── Core iterative loop ──────────────────────────────────────────────────
function pcs_vals = run_iterative(sizes, n_factors, depth, width, ...
                                   beta, best_idx, all_combos, ...
                                   n_reps, batch_size, noise_std, retrain_full)
    pcs_vals = zeros(size(sizes));

    for si = 1:length(sizes)
        target_n = sizes(si);
        correct  = 0;

        for rep = 1:n_reps
            rng(42 + rep);
            X_acc = zeros(0, n_factors);
            y_acc = zeros(0, 1);
            X_new = []; y_new = [];

            for batch_start = 1:batch_size:target_n
                n_new = min(batch_size, target_n - batch_start + 1);
                X_new = double(rand(n_new, n_factors) > 0.5);
                y_new = X_new * beta + randn(n_new, 1) * noise_std;
                X_acc = [X_acc; X_new]; %#ok<AGROW>
                y_acc = [y_acc; y_new]; %#ok<AGROW>
            end

            if retrain_full
                % FIX: always train on everything collected so far
                X_train = X_acc;
                y_train = y_acc;
            else
                % BUG: only the last batch (reproduces original plateau)
                X_train = X_new;
                y_train = y_new;
            end

            predicted = train_and_predict_local(X_train, y_train, ...
                                                depth, width, all_combos);
            if predicted == best_idx
                correct = correct + 1;
            end
        end

        pcs_vals(si) = correct / n_reps;
        fprintf('  n=%4d  PCS=%.3f\n', target_n, pcs_vals(si));
    end
end


%% ── Helpers (duplicated to keep file self-contained) ─────────────────────
function best_idx = train_and_predict_local(X, y, depth, width, all_combos)
    n_factors = size(X, 2);
    lr = 0.01; epochs = 300;
    dims = [n_factors, repmat(width, 1, depth), 1];
    n_layers = length(dims) - 1;
    W = cell(n_layers,1); b = cell(n_layers,1);
    for l = 1:n_layers
        W{l} = randn(dims(l), dims(l+1)) * sqrt(2/dims(l));
        b{l} = zeros(1, dims(l+1));
    end
    for ~1:epochs  % suppress unused-variable warning
    end
    for epoch = 1:epochs
        A = cell(n_layers+1,1); Z = cell(n_layers,1);
        A{1} = X;
        for l = 1:n_layers
            Z{l} = A{l}*W{l} + b{l};
            A{l+1} = (l<n_layers)*max(0,Z{l}) + (l==n_layers)*Z{l};
        end
        delta = 2*(A{end}-y)/size(X,1);
        for l = n_layers:-1:1
            dW = A{l}'*delta; db = sum(delta,1);
            if l>1, delta = (delta*W{l}').*double(Z{l-1}>0); end
            W{l}=W{l}-lr*dW; b{l}=b{l}-lr*db;
        end
    end
    A_p = all_combos;
    for l=1:n_layers
        Zp = A_p*W{l}+b{l};
        A_p = (l<n_layers)*max(0,Zp)+(l==n_layers)*Zp;
    end
    [~,best_idx]=max(A_p);
end

function combos = generate_combinations(n)
    combos = zeros(2^n, n);
    for i = 1:2^n
        combos(i,:) = dec2bin(i-1,n)-'0';
    end
end
