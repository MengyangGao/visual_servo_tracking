function x = dls_solve(A, b, damping)
%SIMULATION_DLS_SOLVE Damped least-squares solution of A*x = b.

if nargin < 3 || isempty(damping)
    damping = 1e-6;
end

A = double(A);
b = double(b(:));
n = size(A, 2);
x = (A.' * A + (damping ^ 2) * eye(n)) \ (A.' * b);
end
