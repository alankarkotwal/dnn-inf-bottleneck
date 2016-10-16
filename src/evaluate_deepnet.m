function evaluate_deepnet(deepnet, xs_test, ys_test)

    types = deepnet(ys_test);
    plotconfusion(xs_test, types);

end