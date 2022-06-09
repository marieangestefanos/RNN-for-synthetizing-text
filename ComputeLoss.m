function L = ComputeLoss(X, Y, RNN, h)

    [A, H, P] = ForwardPass(X, Y, h, RNN);

    n = size(Y, 2);
    lcross = zeros(1, n);
    for i = 1:n
        lcross(i) = Y(:, i)' * log(P(:, i)+eps);
    end
    L = - sum(lcross);

end