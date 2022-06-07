function P = softmax(s)
    K = size(s, 1);
    P = exp(s)./repmat(ones(1, K)*exp(s), [K, 1]);
end