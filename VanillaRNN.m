function final_ind = VanillaRNN(h0, x0, n, RNN)
    
    h = h0;
    x = x0;
    
    K = size(RNN.U, 2);
    %Y = zeros(K, n);
    final_ind = zeros(n, 1);

    for t=1:n
        a = RNN.W * h + RNN.U * x + RNN.b;
        h = tanh(a);
        o = RNN.V * h + RNN.c;
        p = softmax(o);

        cp = cumsum(p);
        j = rand;
        ixs = find(cp-j > 0);
        ii = ixs(1);

        x = indToOneHot(K, ii);
        %Y(:, i) = indToOneHot(ii);
        final_ind(t) = ii;

    end

end