function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

CTries = [0.01 ; 0.03 ; 0.1; 0.3 ; 1 ; 3 ; 10 ; 30];
sigmaTries = [ 0.01; 0.03; 0.1; 0.3; 1; 3 ; 10 ; 30];
model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
predictions = svmPredict (model, Xval);
error = mean(double(predictions ~=yval));

for i = 1:length(CTries)
    for j = 1:length (sigmaTries)
        CTry = CTries(i);
        sigmaTry = sigmaTries(j);
        model= svmTrain(X, y, CTry, @(x1, x2) gaussianKernel(x1, x2, sigmaTry)); 
        predictions = svmPredict (model, Xval);
        errorTry = mean(double(predictions ~= yval));
        if errorTry < error
            error = errorTry;
            C = CTries(i);
            sigma = sigmaTries(j);
        end
    end
end







% =========================================================================

end
