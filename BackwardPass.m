function Grads = BackwardPass(X, Y, A, H, P, RNN)
    n = size(X, 2);
    [K, m] = size(RNN.V);

    G = - ( Y - P )';

    Grads.U = zeros(m, K);
    Grads.V = zeros(K, m);
    Grads.W = zeros(m, m);

    Grads.b = zeros(m, 1);
    Grads.c = zeros(K, 1);

    % Grads h and a
    dLdh = zeros(n, m);
    dLda = zeros(n, m);
    dLdh(n, :) = G(n, :) * RNN.V;
    dLda(n, :) = dLdh(n, :) * diag(1 - tanh(A(:, n)).^2);

    for t=n-1:-1:1
        dLdh(t, :) = G(t, :) * RNN.V + dLda(t+1, :) * RNN.W;
        dLda(t, :) = dLdh(t, :) * diag(1 - tanh(A(:, t)).^2);
    end

    for i = 1:n

        Grads.b = Grads.b + dLda(i, :)';
        Grads.c = Grads.c + G(i, :)';

        Grads.V = Grads.V + G(i, :)' * H(:, i+1)';
        Grads.W = Grads.W + dLda(i, :)' * H(:, i)';
        Grads.U = Grads.U + dLda(i, :)' * X(:, i)';

    end

end