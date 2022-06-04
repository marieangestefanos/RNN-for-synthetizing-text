% Load data
book_fname = 'goblet_book.txt';
fid = fopen(book_fname, 'r');
book_data = fscanf(fid, '%c');
fclose(fid);

% Mapping functions
book_chars = unique(book_data);
K = size(book_chars, 2); %dim of the output (and input here)
char_to_ind = containers.Map('KeyType','char','ValueType','int32');
ind_to_char = containers.Map('KeyType','int32','ValueType','char');

for k = 1:K
    char_to_ind(book_chars(k)) = k;
    ind_to_char(k) = book_chars(k);
end

% Parameters of the network and the training
m = 100; %hidd layer size
eta = 0.1;
seq_length = 25;
% Parameters of the model
RNN.b = zeros(m, 1);
RNN.c = zeros(K, 1);
sig = 0.01;
RNN.U = randn(m, K) * sig;
RNN.W = randn(m, m) * sig;
RNN.V = randn(K, m) * sig;