function [ Error ] = GOSPAmetric(X,Y,c,p)

% Function that computes the Generalised Optimal Subpattern Assignment (GOSPA) Metric
% for the random sets X and Y. Assume that X represents the estimates and Y
% represents the ground truth

% Input X (and Y) is with J estimates
% X -- Nx * J matrix with kinematic vectors

% Number of elements in X and Y.
Nx = size(X,2);
Ny = size(Y,2);

if Nx>Ny % assume X contains fewer elements than Y
    tmp = X;
    X = Y;
    Y = tmp;
    Nx = size(X,2);
    Ny = size(Y,2);
end

% if there are no elements at all
if Nx==0 && Ny==0
    gospa = 0;
    LocationError = 0;
    MissedError = 0;
    FalseError = 0;
    Error = [gospa,LocationError,MissedError,FalseError];
    return
elseif Nx==0
    gospa = (0.5*(c^p)*Ny)^(1/p);
    LocationError = 0;
    MissedError = Ny;
    FalseError = 0;
    Error = [gospa,LocationError,MissedError,FalseError];
    return
elseif Ny==0
    gospa = (0.5*(c^p)*Nx)^(1/p);
    LocationError = 0;
    MissedError = 0;
    FalseError = Nx;
    Error = [gospa,LocationError,MissedError,FalseError];
    return
end

% Distance matrix
D = repmat(c,[Nx Ny]);

for ix = 1:Nx
    for iy = 1:Ny
        % Euclidean Distance for kinematics
        gwd = norm(X(:,ix)-Y(:,iy));
        
        % Apply threshold c
        D(ix,iy) = gwd;
    end
end

% Allocate memory
gospa = 0;
LocationError = 0;
absCard = 0;

% Compute assignment
[Customer2Item,~] = assign2D(D);

% Iterate over estimates
for ix = 1:Nx
    % Check if distance is small enough
    if D(ix,Customer2Item(ix)) < c
        % Location part of GOSPA
        gospa = gospa + D(ix,Customer2Item(ix))^p;
        % Location error
        LocationError = LocationError + D(ix,Customer2Item(ix));
        % Number of assignments
        absCard = absCard + 1;
    end
end

gospa = (gospa + 0.5*(c^p)*(Nx+Ny-2*absCard))^(1/p);

% Missed detection error
MissedError = Ny-absCard;

% False alarm error
FalseError = Nx-absCard;

Error = [gospa,LocationError/absCard,MissedError,FalseError];


