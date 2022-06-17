function [RNN, smooth_losses] = SGD(RNN, SGDParams)

    epoch = 1;    
    e = 1;
    smooth_losses = [];
    ite = 0;
    [m, K] = size(RNN.U);
    
    for f = fieldnames(RNN)'
        momentum.(f{1}) = 0;
    end
    
    while ite <= 100000
        if e == 1
            hprev = zeros(m, 1);
        else
            hprev = H(:, end);
        end
        
        % Forward pass
        
        X_chars = SGDParams.book_data(e:e+SGDParams.seq_length-1);
        Y_chars = SGDParams.book_data(e+1:e+SGDParams.seq_length);
        
        X = ToOneHotArray(X_chars, K, SGDParams.char_to_ind);
        Y = ToOneHotArray(Y_chars, K, SGDParams.char_to_ind);
        
        [A, H, P, L] = ForwardPass(X, Y, hprev, RNN);

        
        % Backward pass (Gradient Update with Adagrad)
        Grads = BackwardPass(X, Y, A, H, P, RNN);
        for f = fieldnames(Grads)'
            momentum.(f{1}) = momentum.(f{1}) + Grads.(f{1}).^2;
            eta = SGDParams.eta ./ sqrt(momentum.(f{1}) + eps);
            RNN.(f{1}) = RNN.(f{1}) - eta .* Grads.(f{1});
        end
        
        % Computing smooth loss
        if e == 1 && epoch == 1
            smooth_losses(end+1) = L;
            bestLoss = L;
        else
            smooth_losses(end+1) = 0.999 * smooth_losses(end) + 0.001*L;
            if smooth_losses(end) < bestLoss
                bestLoss = smooth_losses(end);
                bestRNN = RNN;
                bestHprev = hprev;
            end
        end

        % Display

        if mod(ite, 10000) == 0
            epoch
            ite
            smooth_losses(end)
        end

        if mod(ite, 10000) == 0
            txt = "";
            seq = VanillaRNN(hprev, X(:, 1), 200, RNN);
            for char=seq
                txt = txt + SGDParams.ind_to_char(char);
            end
            disp(txt);
        end
        
        ite = ite + 1;
        e = e + SGDParams.seq_length;
        if e > length(SGDParams.book_data)-SGDParams.seq_length-1
            e = 1;
            epoch = epoch + 1;
        end

    end

    txt = "";
    seq = VanillaRNN(bestHprev, X(:, 1), 1000, bestRNN);
    for char=seq
        txt = txt + SGDParams.ind_to_char(char);
    end
    disp(txt);
    
    disp(bestLoss);



end