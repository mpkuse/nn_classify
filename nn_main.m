%% Learn-Test 1-layer hidden neural network for 3 category classification
clear all;
close all;
%% Generate Data / Load Data
if false
    display( 'Generate Data' );
    nClass = 3;
    dataPerClass = 100;
    dim = 2;
    [D Loss L_nn] = spiral_data( dataPerClass, nClass ); % 100 points of each category. 3 categories in all
    nTotalD = nClass*dataPerClass;
    save( 'data.mat', 'nClass', 'dataPerClass', 'dim', 'D', 'L', 'L_nn', 'nTotalD' )
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
h = 2; % # of hidden nodes
W1 = randn( dim, h );
b1 = randn( 1, h );
W2 = randn( h, nClass );
b2 = randn( 1, nClass );


%% Iterations
step = 0.004;
lambda = .1;
nItr = 100;
sgd_pick = 240; 
display( 'Iterations Begin...' );
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
        
        [ u1, u2, u3, u4, u5, Loss ] = forward_pass( X, W1, b1, W2, b2, y );        
        [ dL_dW1, dL_db1, dL_dW2, dL_db2 ] = backward_pass( X, W1, b1, W2, b2, y,   u1, u2, u3, u4, u5, Loss );
        
        a_dL_dW1 = a_dL_dW1 + dL_dW1;
        a_dL_db1 = a_dL_db1 + dL_db1;
        a_dL_dW2 = a_dL_dW2 + dL_dW2;
        a_dL_db2 = a_dL_db2 + dL_db2;
        a_cost = a_cost + Loss;
    end
    
    % gradient descent (update params)
    W1 = W1 - step * (a_dL_dW1 + lambda*W1);
    b1 = b1 - step * (a_dL_db1 + lambda*b1);
    W2 = W2 - step * (a_dL_dW2 + lambda*W2);
    b2 = b2 - step * (a_dL_db2 + lambda*b2);
    
    
    % Eval accuracy of currect weights
    if mod(itr,15 ) == 0 %eval accuracy every 15 iterations
        [acc ] = eval_perf( D_test, L_nn_test, W1, b1, W2, b2 );
        display( sprintf( '%d : %f (acc=%f)', itr, a_cost, acc ) );
    end

end

%% Generate Summary
display( '--------------\nSummary\n--------------' );
%display( 'Test' );
%eval_perf_detail( D_test, L_nn_test, W1, b1, W2, b2 );
%display( 'Train' );
%eval_perf_detail( D_train, L_nn_train, W1, b1, W2, b2 );
display( 'Train+Test' );
eval_perf_detail( D, L_nn, W1, b1, W2, b2 );

%% Understand these mappings by hidden layer
H1 = max( 0, D*W1 + repmat( b1, size(D,1), 1 ) ); %output of 1st hidden layer
class_1 = L==1;
class_2 = L==2;
class_3 = L==3;
plot( H1(class_1,1), H1(class_1,2), 'r.' ), title( 'output of 1st hidden layer' );
hold on
plot( H1(class_2,1), H1(class_2,2), 'g.' );
plot( H1(class_3,1), H1(class_3,2), 'b.' );
