function oneHotArray = ToOneHotArray(array, K, char_to_ind)
    n = size(array, 2);
    oneHotArray = zeros(K, n);
    for i = 1:n
        oneHotArray(char_to_ind(array(1, i)), i) = 1;
    end
end