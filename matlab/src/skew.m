function S = skew(v)
%SIMULATION_SKEW Return the skew-symmetric matrix of a 3-vector.

v = v(:);
S = [   0   -v(3)  v(2);
      v(3)    0   -v(1);
     -v(2)  v(1)    0 ];
end
