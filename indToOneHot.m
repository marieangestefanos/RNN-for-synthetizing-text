function oneHot = indToOneHot(K, ind)
    oneHot = zeros(K, 1);
    oneHot(ind) = 1;
end