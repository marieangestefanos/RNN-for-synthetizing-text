function [A, H, P, loss] = ForwardPass(X, Y, h0, RNN)

    n = size(X, 2);
    [K, m] = size(RNN.V);

    A = zeros(m, n);
    H = zeros(m, n+1);
    H(:, 1) = h0;
    P = zeros(K, n);

    loss = 0;

    for t=1:n 
        A(:, t) = RNN.W * H(:, t) + RNN.U * X(:, t) + RNN.b;
        H(:, t+1) = tanh(A(:, t));
        o = RNN.V * H(:, t+1) + RNN.c;
        P(:, t) = softmax(o);

        loss = loss - Y(:, t)' * log(P(:, t)+eps);
    end

end