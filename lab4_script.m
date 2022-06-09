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

%rng(400);

% Parameters of the network and the training
m = 5; %hidd layer size
eta = 0.1;
seq_length = 25;
% Parameters of the model
RNN.b = zeros(m, 1);
RNN.c = zeros(K, 1);
sig = 0.01;
RNN.U = randn(m, K) * sig;
RNN.W = randn(m, m) * sig;
RNN.V = randn(K, m) * sig;


% Synthesize text
% h0 = zeros(m, 1); 
% x0 = zeros(K, 1);
% final_ind = VanillaRNN(h0, x0, seq_length, RNN);

% final_seq = [];
% for i = 1:seq_length
%     final_seq = strcat(final_seq, ind_to_char(final_ind(i)));
% end
% final_seq

% Forward and backward pass of back_prop

X_chars = book_data(1:seq_length);
Y_chars = book_data(2:seq_length+1);

X = ToOneHotArray(X_chars, K, char_to_ind);
Y = ToOneHotArray(Y_chars, K, char_to_ind);

h0 = zeros(m, 1); 

[A, H, P] = ForwardPass(X, Y, h0, RNN);
GradsAn = BackwardPass(X, Y, A, H, P, RNN);

%%

% Gradients comparisons
GradsNum = ComputeGradsNum(X, Y, RNN, 1e-4);
errors = ComputeRelativeError(GradsAn, GradsNum);

%%
i = 1;
for f = fieldnames(GradsAn)'
    sprintf('Error on %s', f{1})
    errors(i)
    i = i+1;
end