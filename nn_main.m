%% Learn-Test 1-layer hidden neural network for 3 category classification
clear all;
close all;
%% Generate Data
nClass = 3;
dataPerClass = 100;
dim = 2;
[D L L_nn] = spiral_data( 100, 3 ); % 100 points of each category. 3 categories in all
nTotalD = nClass*dataPerClass;

%% init
h = 5; % # of hidden nodes
W1 = randn( dim, h );
b1 = randn( 1, h );
W2 = randn( h, nClass );
b2 = randn( 1, nClass );


%% Iterations
step = 0.001
for itr = 1:600
    
    a_dL_dW1 = zeros( dim, h );
    a_dL_db1 = zeros( 1, h );
    a_dL_dW2 = zeros( h, nClass );
    a_dL_db2 = zeros( 1, nClass );
    a_cost = 0;
    
    % going over data to compute gradient and total cost function
    % TODO for speed should actually go over random indices only (SGD)
    for e=1:size(D,1) 
        X = D( e, : );
        y = L_nn( e, :);
        
        [ u1, u2, u3, u4, u5, L ] = forward_pass( X, W1, b1, W2, b2, y );        
        [ dL_dW1, dL_db1, dL_dW2, dL_db2 ] = backward_pass( X, W1, b1, W2, b2, y,   u1, u2, u3, u4, u5, L );
        
        a_dL_dW1 = a_dL_dW1 + dL_dW1;
        a_dL_db1 = a_dL_db1 + dL_db1;
        a_dL_dW2 = a_dL_dW2 + dL_dW2;
        a_dL_db2 = a_dL_db2 + dL_db2;
        a_cost = a_cost + L;
    end
    
    % gradient descent (update params)
    W1 = W1 - step * a_dL_dW1;
    b1 = b1 - step * a_dL_db1;
    W2 = W2 - step * a_dL_dW2;
    b2 = b2 - step * a_dL_db2;
    
    display( sprintf( '%d : %f', itr, a_cost ) );
end