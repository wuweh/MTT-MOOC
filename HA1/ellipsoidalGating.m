function z_ingate = ellipsoidalGating(x, P, z, measmodel, gating_size)
%GATING performs ellipsoidal gating for a single target
%INPUT:  z: measurements --- (measurement dimension) x (number
%           of measurements) matrix
%OUTPUT: z_ingate: measurements in the gate --- (measurement
%                   dimension) x (number of measurements in the gate) matrix
zlength = size(z,2);
in_gate = false(zlength,1);

S = measmodel.H(x)*P*measmodel.H(x)' + measmodel.R;
%Make sure matrix S is positive definite
S = (S+S')/2;
%Use choleskey decomposition to speed up matrix decomposition
[Vs,~] = chol(S);

nu = z - repmat(measmodel.h(x),[1 zlength]);
dist= sum((inv(Vs)'*nu).^2);

in_gate(dist<gating_size) = true;
z_ingate = z(:,in_gate);

end