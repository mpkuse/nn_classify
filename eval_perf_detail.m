function [  ] = eval_perf_detail( D_test, y_test, W1, b1, W2, b2 )
% Given the trainned weights, test data set and its labels, evaluates
% performance of this network

correct = 0;
out_stack = [];
correctness_stack = zeros( size(D_test, 1), 1 ); % if outcome and GT match this is set to 1
confusion_matrix = zeros( size(y_test,2) );

for i=1:size(D_test,1)
    [ outcome ] = forward_pass_predict( D_test(i,:), W1, b1, W2, b2 );
    out_stack = cat(1, out_stack, outcome );
    
    if sum( abs(y_test( i, : ) - outcome) ) == 0
        correct = correct + 1;
        correctness_stack(i) = 1;
    end
    
    %confusion matrix
    [C_gt I_gt] = max( y_test( i, : ) );
    [C_prd I_prd] = max( outcome );
    confusion_matrix( I_gt, I_prd ) =  confusion_matrix( I_gt, I_prd ) + 1;
end

accuracy = correct / size(D_test,1);
display( sprintf( 'Final Accuracy : %f', accuracy ) );
confusion_matrix


correct_indx = correctness_stack == 1;
wrong_indx = correctness_stack == 0;

figure, plot( D_test( correct_indx, 1 ), D_test( correct_indx, 2 ), 'b.' ), title( 'correct predictions in BLUE, wrong predictions in RED' );
hold on
plot( D_test( wrong_indx, 1 ), D_test( wrong_indx, 2 ), 'r.' );

end

