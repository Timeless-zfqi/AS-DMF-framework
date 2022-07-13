function [ dis ] = manhattanDist( X, Y )
%MAHATTENDIST Summary of this function goes here
%   Compute the mahatten distance between X and Y for each row.
%   Input: X, Y, two row vector matrix
%   Output: If X has n1 rows and Y has n2 rows return matrix with n1*n2 scale.
%   
    
    [n1, d1] = size(X);
    [n2, d2] = size(Y);
    if d1 ~= d2
        error('the number of columns must be equal with each other');
    end
    dis = zeros(n1, n2);
    for i = 1:n1
        for j = 1:n2
            dis(i, j) = sum(abs(X(i, :) - Y(j, :)));
        end
    end
end

