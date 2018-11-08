function z_ingate = Gating(x, P, z, measmodel, gating_size)
%GATING performs ellipsoidal gating for a single target
%INPUT:  z: measurements --- (measurement dimension) x (number
%           of measurements) matrix
%OUTPUT: z_ingate: measurements in the gate --- (measurement
%                   dimension) x (number of measurements in the gate) matrix
zlength = size(z,2);
in_gate = false(zlength,1);
S = measmodel.H*P*measmodel.H' + measmodel.R;
nu = z - measmodel.H*repmat(x,[1 zlength]);
dist= sum((inv(chol(S))'*nu).^2);
in_gate(dist<gating_size) = true;
z_ingate = z(:,in_gate);
end