function [xs_train, ys_train, xs_test, ys_test] = ...
    generate_data(n_categories, n_features, n_samples, n_test_samples)

    % We consider a classification task with features drawn from a 
    % Gaussian mixture model

    rng('default');

    % n_categories = 5;
    % n_features = 10;
    % n_samples = 1000;
    % n_test_samples = 100;

    mus = 2*randn(n_categories, n_features);
    sigma = 10*eye(n_features);

    model = gmdistribution(mus, sigma);

    ys_train = zeros(n_features, n_samples);
    xs_train = zeros(n_categories, n_samples);

    for i = 1:n_samples

        [y, x] = random(model);
        xs_train(x, i) = 1;
        ys_train(:, i) = y;

    end

    ys_test = zeros(n_features, n_test_samples);
    xs_test = zeros(n_categories, n_test_samples);

    for i = 1:n_test_samples

        [y, x] = random(model);
        xs_test(x, i) = 1;
        ys_test(:, i) = y;

    end

end