function [Data Label, Label_nn] = spiral_data(nDataPts, nClass)
%
% Generate artificial spiral data in 2-d
%

%nDataPts = 100; %# of pts in each category
%nClass = 3;


noise =  .2*randn( 1, nDataPts );
colors = [ 'r.'; 'g.'; 'b.' ; 'k.' ; 'r*'; 'g*'; 'b*' ; 'k*' ];

Data = [];
Label = [];
Label_nn = [];
for k=1:nClass
    r = linspace( 0, 1, nDataPts );
    t = linspace( 2*pi/nClass*k, 2*pi/nClass*(k+1), nDataPts );

    x1 = r .* cos(t+noise);
    x2 = r .* sin(t+noise);
    label = k * ones(1, nDataPts );
    label_nn = zeros( 1, nClass );
    label_nn(k) = 1;

    plot( x1,x2, colors(k,:) );
    axis( [-1, 1, -1, 1] );
    hold on
    
    Data = cat( 1, Data, [ x1' x2' ] );
    Label = cat( 1, Label, label' );
    Label_nn = cat( 1, Label_nn, repmat( label_nn, nDataPts, 1 ) );
end

clear colors k label noise r t x1 x2;