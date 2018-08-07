function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in thetah_

% determine hypothesis
h_theta = sigmoid(X * theta);

% determine cost function
t1 = -y .* log(h_theta);
t2 = (1-y) .* log(1 - h_theta);

% note first theta is not reguralized
t3 = lambda/(2*m) * sum(theta(2:end).^2);
J = 1/m * sum(t1 - t2) + t3;

% determine gradient for first term
grad(1) = 1/m * sum((h_theta - y).*X(:, 1));

% determine gradient for the other parameters
grad(2:end) = 1/m * sum((h_theta - y).*X(:, 2:end))' + lambda/m *theta(2:end);
% =============================================================

end
