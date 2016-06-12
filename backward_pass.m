function [ dL_dW1, dL_db1, dL_dW2, dL_db2 ] = backward_pass( X, W1, b1, W2, b2, y,   u1, u2, u3, u4, u5, L )

h = length(b1);
% gradient of softmax
% see derivation : http://math.stackexchange.com/questions/945871/derivative-of-softmax-loss-function
exp_u = exp( u5 );
sigma = sum( exp_u );
p = exp_u / sigma;
dL_du5 = p - y; %1x3 (mayb transpose)

du5_db2 = eye(3); %3x3
du5_du4 = eye(3); %3x3

du4_du3 = W2'; %3xh
du4_dW2 = u3;  %1xh

% if u2_i > 0 ==> 1. if u2_i < 0 ==> 0
du3_du2 = diag(max( 0, u2 ) > 0); %hxh

du2_db1 = eye(h); %hxh
du2_du1 = eye(h); %hxh

du1_dW1 = X; %1xd


% chaining
dL_db2 = dL_du5 * du5_db2;
dL_dW2 = du4_dW2'   *    dL_du5 * du5_du4;

dL_db1 = dL_du5 * du5_du4 * du4_du3 * du3_du2 * du2_db1;

dL_dW1 = du1_dW1'    *    dL_du5 * du5_du4 * du4_du3 * du3_du2 * du2_du1 ;

end

