%% Learn-Test 1-layer hidden neural network for 3 category classification
clear all;
close all;
%% Generate Data / Load Data
if false
    display( 'Generate Data' );
    nClass = 3;
    dataPerClass = 100;
    dim = 2;
    [D L L_nn] = spiral_data( 100, 3 ); % 100 points of each category. 3 categories in all
    nTotalD = nClass*dataPerClass;
    %save( 'data.mat', 'nClass', 'dataPerClass', 'dim', 'D', 'L', 'L_nn', 'nTotalD' )
else
    display( 'Load Data' );
    load( 'data.mat' );
end



%% Separate trainning set and test set
nD = dataPerClass; %just an alias
r_perm = randperm( nD );
frac_train = 0.8; %fraction of data to be used for training

fl = floor(frac_train*nD);
tr_p = r_perm( 1:fl ); % train_permutation %TODO 0.8*dataPerClass
te_p  = r_perm( (fl+1):end ); % test permutation
display( sprintf( 'Using %d points for training and %d points for testing', length(tr_p), length(te_p) ) );

D_train = cat(1, D( tr_p, : ), D( nD+tr_p, : ), D( nD*2 + tr_p, : ) );
L_nn_train = cat(1, L_nn( tr_p, : ), L_nn( nD+tr_p, : ), L_nn( nD*2 + tr_p, : ) );

D_test = cat( 1, D( te_p, : ), D( nD+te_p, : ), D( nD*2+te_p, : ) );
L_nn_test = cat( 1, L_nn( te_p, :), L_nn( nD+te_p, :), L_nn( nD*2+te_p, :) );

%% Init - Network
h = 7; % # of hidden nodes
W1 = randn( dim, h );
b1 = randn( 1, h );
W2 = randn( h, nClass );
b2 = randn( 1, nClass );


%% Iterations
step = 0.004;
lambda = .1;
nItr = 100;
sgd_pick = 180; 

for itr = 1:nItr
    
    a_dL_dW1 = zeros( dim, h );
    a_dL_db1 = zeros( 1, h );
    a_dL_dW2 = zeros( h, nClass );
    a_dL_db2 = zeros( 1, nClass );
    a_cost = 0;
    
    % going over data to compute gradient and total cost function

    % SGD - Stocastic Gradient
    sgd_perm = randperm(   min(size(D_train,1),sgd_pick)  ); %pick 10
    D_sgd = D_train( sgd_perm, : );
    L_nn_sgd = L_nn_train( sgd_perm, : );
    
    for e=1:size(D_sgd,1) 
        X = D_sgd( e, : );
        y = L_nn_sgd( e, :);
        
        [ u1, u2, u3, u4, u5, L ] = forward_pass( X, W1, b1, W2, b2, y );        
        [ dL_dW1, dL_db1, dL_dW2, dL_db2 ] = backward_pass( X, W1, b1, W2, b2, y,   u1, u2, u3, u4, u5, L );
        
        a_dL_dW1 = a_dL_dW1 + dL_dW1;
        a_dL_db1 = a_dL_db1 + dL_db1;
        a_dL_dW2 = a_dL_dW2 + dL_dW2;
        a_dL_db2 = a_dL_db2 + dL_db2;
        a_cost = a_cost + L;
    end
    
    % gradient descent (update params)
    W1 = W1 - step * (a_dL_dW1 + lambda*W1);
    b1 = b1 - step * (a_dL_db1 + lambda*b1);
    W2 = W2 - step * (a_dL_dW2 + lambda*W2);
    b2 = b2 - step * (a_dL_db2 + lambda*b2);
    
    
    % Eval accuracy of currect weights
    if mod(itr,15 ) == 0 %eval accuracy every 15 iterations
        [acc confusion_mat out_stack] = eval_perf( D_test, L_nn_test, W1, b1, W2, b2 );
        display( sprintf( '%d : %f (acc=%f)', itr, a_cost, acc ) );
    end

end
