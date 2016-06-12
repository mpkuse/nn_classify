%% Learn-Test 1-layer hidden neural network for 3 category classification

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

X = D( 10, : );
y = L_nn( 10, :);
[ u1, u2, u3, u4, u5, L ] = forward_pass( X, W1, b1, W2, b2, y );