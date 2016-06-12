function [ u1, u2, u3, u4, u5, L ] = forward_pass( X, W1, b1, W2, b2, y )

u1 = X * W1;
u2 = u1 + b1;
u3 = max( 0, u2 );
u4 = u3 * W2;
u5 = u4 + b2;

%u5 and y must be of same size
L = SoftMaxLoss( u5, y );

end

