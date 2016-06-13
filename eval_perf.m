function [ accuracy, confusion_matrix, out_stack ] = eval_perf( D_test, y_test, W1, b1, W2, b2 )
% Given the trainned weights, test data set and its labels, evaluates
% performance of this network

correct = 0;
out_stack = [];
confusion_matrix = zeros( size(y_test,2) );

for i=1:size(D_test,1)
    [ outcome ] = forward_pass_predict( D_test(i,:), W1, b1, W2, b2 );
    out_stack = cat(1, out_stack, outcome );
    
    if sum( abs(y_test( i, : ) - outcome) ) == 0
        correct = correct + 1;
    end
    
    %confusion matrix
    [C_gt I_gt] = max( y_test( i, : ) );
    [C_prd I_prd] = max( outcome );
    confusion_matrix( I_gt, I_prd ) =  confusion_matrix( I_gt, I_prd ) + 1;
end

accuracy = correct / size(D_test,1);

end

