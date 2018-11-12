function z_ingate = Gating(x, P, z, measmodel, gating_size)
%GATING performs ellipsoidal gating for a single target
%INPUT:  z: measurements --- (measurement dimension) x (number
%           of measurements) matrix
%OUTPUT: z_ingate: measurements in the gate --- (measurement
%                   dimension) x (number of measurements in the gate) matrix
zlength = size(z,2);
in_gate = false(zlength,1);
S = measmodel.H(x)*P*measmodel.H(x)' + measmodel.R;
S = (S+S')/2;
nu = z - repmat(measmodel.h(x),[1 zlength]);
[Vs,~] = chol(S);
dist= sum((inv(Vs)'*nu).^2);
in_gate(dist<gating_size) = true;
z_ingate = z(:,in_gate);
% if p == 1
%     z_ingate = [];
% else
%     dist= sum((inv(Vs)'*nu).^2);
%     in_gate(dist<gating_size) = true;
%     z_ingate = z(:,in_gate);
% end
end