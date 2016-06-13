function [ outcome ] = forward_pass_predict( X, W1, b1, W2, b2 )

u1 = X * W1;
u2 = u1 + b1;
u3 = max( 0, u2 );
u4 = u3 * W2;
u5 = u4 + b2;

%u5 and y must be of same size
%L = SoftMaxLoss( u5, y );

%exp_u = exp( u5 );
%sigma = sum( exp_u );

%p = -log( exp_u / sigma );


[V VV] = max( u5 );
outcome = zeros( size(u5) );
outcome(VV) = 1;

end

