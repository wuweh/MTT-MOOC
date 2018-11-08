function root_mean_square_error = RMSE(x1,x2)
%RMSE calculates the root mean square error between x and x_hat
%INPUT: x1: (variable dimension) x (number of variables) matrix
%       x2: (variable dimension) x (number of variables) matrix
%OUTPUT: root_mean_square_error: (variable dimension) x 1 vector
root_mean_square_error = sqrt(mean((x1-x2).^2,2));

end

