function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    % Add % to toggle between both solutions below

    % Un-Vectorized
    % v1 = zeros(m, 1);
    % v2 = zeros(m, 1);
    % for i = 1:m
    %     v1(i) = ((theta(1)+theta(2)*X(i,2))-y(i)) ;
    %     v2(i) = ((theta(1)+theta(2)*X(i,2))-y(i))*X(i,2) ;
    % end
    % theta(1) = theta(1) - (alpha/m)*sum(v1);
    % theta(2) = theta(2) - (alpha/m)*sum(v2);

    % Vectorized
    j = 1:m;
    t1 = sum((theta(1)+theta(2).*X(j,2))-y(j));
    t2 = sum(((theta(1)+theta(2).*X(j,2))-y(j)).* X(j,2));    
    theta(1) = theta(1)-(alpha/m)*(t1);
    theta(2) = theta(2)-(alpha/m)*(t2);




    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
