function nearest_neighbor_meas = nearestNeighbor(z, meas_likelihood)
%NEARESTNEIGHBOR finds the measurement with the highest
%measurement update likelihood
%INPUT:  z: measurements --- (measurement dimension) x (number
%of measurements) matrix
%        meas_likelihood: measurement update likelihood for
%each measurement --- (measurement dimension) x 1 vector
%OUTPUT: nearest_neighbor_meas --- single measurement of size
%(measurement dimension) x 1
[~, nearest_neighbor_assoc] = max(meas_likelihood);
nearest_neighbor_meas = z(:,nearest_neighbor_assoc);
end