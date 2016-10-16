function deepnet = train_network(arch, xs_train, ys_train)

    features_old = ys_train;
    autoencs = cell(size(arch, 1), 1);

    for i = 1:size(arch, 1)
        
        hiddenSize = arch(i);
        autoenc = trainAutoencoder(features_old, hiddenSize, ...
            'L2WeightRegularization', 0.001, ...
            'SparsityRegularization', 4, ...
            'SparsityProportion', 0.05, ...
            'DecoderTransferFunction', 'purelin', ...
            'ShowProgressWindow', 0);
        autoencs{i} = autoenc;
        features_old = encode(autoenc, features_old);
        
    end
    
    softnet = trainSoftmaxLayer(features_old, xs_train, ...
              'LossFunction', 'crossentropy', 'ShowProgressWindow', 0);
    deepnet = stack(autoencs{:}, softnet);
    deepnet = train(deepnet, ys_train, xs_train);

end