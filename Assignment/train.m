function images = train(train_set,train_label,test_set,test_label)

% training
tic; 
model = svmtrain(train_label, train_set, '-s 0 -t 0');
t1 = toc;
% classification
tic;
[predicted_label, accuracy, decision_values]=svmpredict(test_label, test_set, model);
t2 = toc;
disp(num2str(t1));
disp(num2str(t2));
end