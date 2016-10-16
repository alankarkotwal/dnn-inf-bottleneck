% Yayyyyyyyyyyy

[xs_train, ys_train, xs_test, ys_test] = ...
    generate_data(5, 10, 10000, 1000);

arch = [9; 2];
deepnet = train_network(arch, xs_train, ys_train);
evaluate_deepnet(deepnet, xs_test, ys_test);