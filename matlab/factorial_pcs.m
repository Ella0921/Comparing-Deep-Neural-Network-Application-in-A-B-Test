%% factorial_pcs.m
% Decentralised Method: Orthogonal Array Designs + Linear OLS Estimator
%
% Computes PCS for full / half / partial factorial designs and
% plots a comparison bar chart.
%
% Usage:
%   factorial_pcs()
%   factorial_pcs(8)   % with 8 factors

function factorial_pcs(n_factors)
    if nargin < 1, n_factors = 5; end

    rng(0);
    beta      = rand(n_factors, 1) * 2 - 1;
    noise_std = 0.5;
    n_reps    = 500;

    all_combos = generate_combinations(n_factors);
    true_resp  = all_combos * beta;
    [~, best_idx] = max(true_resp);

    fractions = {'partial', 'half', 'full'};
    pcs_vals  = zeros(1, 3);
    n_runs    = zeros(1, 3);

    fprintf('%-10s %-8s %-8s\n', 'Design', 'n_runs', 'PCS');
    fprintf('%s\n', repmat('-', 1, 28));

    for fi = 1:length(fractions)
        frac   = fractions{fi};
        design = get_oa_design(n_factors, frac);
        n_runs(fi) = size(design, 1);
        correct    = 0;

        for rep = 1:n_reps
            rng(42 + rep);
            y = design * beta + randn(n_runs(fi), 1) * noise_std;
            predicted = ols_predict_best(design, y, all_combos);
            if predicted == best_idx
                correct = correct + 1;
            end
        end

        pcs_vals(fi) = correct / n_reps;
        fprintf('%-10s %-8d %-8.3f\n', frac, n_runs(fi), pcs_vals(fi));
    end

    % ── Bar chart ─────────────────────────────────────────────────────────
    colors = [0.898 0.325 0.302;   % partial — red
              1.000 0.561 0.000;   % half    — amber
              0.180 0.490 0.196];  % full    — green

    figure('Name', 'Factorial PCS Comparison', 'Position', [100 100 700 420]);
    b = bar(pcs_vals, 'FaceColor', 'flat');
    b.CData = colors;

    set(gca, 'XTickLabel', ...
        {sprintf('Partial\n(n=%d)', n_runs(1)), ...
         sprintf('Half\n(n=%d)', n_runs(2)), ...
         sprintf('Full\n(n=%d)', n_runs(3))}, ...
        'FontSize', 11);
    ylabel('PCS', 'FontSize', 12);
    title(sprintf('Factorial Design PCS — %d factors', n_factors), 'FontSize', 13);
    ylim([0 1.1]);

    % Value labels on bars
    for i = 1:3
        text(i, pcs_vals(i) + 0.03, sprintf('%.3f', pcs_vals(i)), ...
             'HorizontalAlignment', 'center', 'FontSize', 11, 'FontWeight', 'bold');
    end

    grid on;
    saveas(gcf, fullfile('results', 'matlab_fig_factorial.png'));
    fprintf('Saved → results/matlab_fig_factorial.png\n');
end


%% ── OLS: predict best combination from OA observations ──────────────────
function best_idx = ols_predict_best(design, y, all_combos)
    n = size(design, 1);
    X_aug = [ones(n, 1), design];
    coef  = X_aug \ y;                     % least-squares
    preds = [ones(size(all_combos,1),1), all_combos] * coef;
    [~, best_idx] = max(preds);
end


%% ── Orthogonal Array construction via Hadamard ───────────────────────────
function design = get_oa_design(n_factors, fraction)
    switch fraction
        case 'full'
            design = generate_combinations(n_factors);
            return
        case 'half'
            n_runs = 2^(n_factors - 1);
        case 'partial'
            p = 1;
            while (2^p - 1) < n_factors, p = p + 1; end
            n_runs = 2^p;
        otherwise
            error('Unknown fraction: %s', fraction);
    end
    H  = hadamard_matrix(n_runs);
    oa = (H(:, 2:end) + 1) / 2;           % ±1 → 0/1, drop intercept column
    if size(oa, 2) < n_factors
        error('OA too small for %d factors with %d runs', n_factors, n_runs);
    end
    design = oa(:, 1:n_factors);
end

function H = hadamard_matrix(n)
    if n == 1, H = 1; return; end
    H2 = hadamard_matrix(n/2);
    H  = [H2, H2; H2, -H2];
end

function combos = generate_combinations(n)
    combos = zeros(2^n, n);
    for i = 1:2^n
        combos(i, :) = dec2bin(i-1, n) - '0';
    end
end
