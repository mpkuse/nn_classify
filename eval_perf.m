function [ accuracy ] = eval_perf( D_test, y_test, W1, b1, W2, b2 )
% Given the trainned weights, test data set and its labels, evaluates
% performance of this network

correct = 0;
out_stack = [];
for i=1:size(D_test,1)
    [ outcome ] = forward_pass_predict( D_test(i,:), W1, b1, W2, b2 );
    out_stack = cat(1, out_stack, outcome );
    if sum( abs(y_test( i, : ) - outcome) ) == 0
        correct = correct + 1;
    end
end

accuracy = correct / size(D_test,1);

end

