function Ad = adj_transform(T)
%SIMULATION_ADJ_TRANSFORM Adjoint matrix for twists in [v; w] ordering.

R = T(1:3, 1:3);
p = T(1:3, 4);
Ad = [R, skew(p) * R;
      zeros(3, 3), R];
end
