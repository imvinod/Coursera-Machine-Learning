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
%               derivatives of the cost w.r.t. each parameter in theta

red = zeros(m);

hyp = sigmoid(X*theta);

red = (-1 * y') * (log(hyp));
blue = (1 - y') * log(1 - hyp);

theta(1) = 0;
thetasquare = theta' * theta;

J = 1/m * (red - blue) + (lambda/(2 * m))*thetasquare;


	grad(1) = 1/ m * X(:,1)' * (hyp-y);

for i=2:size(theta)

	grad(i) = 1/ m * X(:,i)' * (hyp-y) + lambda/m * theta(i);

end



% =============================================================

end
