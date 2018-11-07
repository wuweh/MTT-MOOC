classdef singletargetracker
    %SINGLETARGETTRACKER is a class containing functions to track a single
    %target in clutter and missed detection.
    
    properties
        gating  % specify gating parameter
        x       % target state mean --- (target state dimension) x 1 vector
        P       % target state covariance --- (target state dimension) x (target state dimension) matrix
    end
    
    methods
        
        function obj = initiator(obj,P_G,m_d,x_0,P_0)
            %INITIATOR initiates singletargetracker class
            %INPUT: P_G: gating size in percentage --- scalar
            %       m_d: measurement dimension --- scalar
            %       x_0: mean of target initial state --- (target state
            %       dimension) x 1 vector
            %       P_0: covariance of target initial state (target state
            %       dimension) x (target state dimension) matrix
            %OUTPUT:  obj.gating.size: gating size --- scalar
            obj.gating.P_G = P_G;
            obj.gating.size = chi2inv(P_G,m_d);
            obj.x = x_0;
            obj.P = P_0;
        end
        
        function obj = linearKalmanPredict(obj, motionmodel)
            %LINEARKALMANPREDICT performs linear Kalman prediction step
            obj.x = motionmodel.A*obj.x;
            obj.P = motionmodel.Q+motionmodel.A*obj.P*motionmodel.A';
        end
        
        function obj = linearKalmanUpdate(obj, measmodel)
            %LINEARKALMANUPDATE performs linear Kalman update step
            S = measmodel.H*obj.P*measmodel.H' + measmodel.R;
            K = obj.P*measmodel.H'/S;
            obj.x = obj.x + K*(z - measmodel.H*obj.x);
            obj.P = (eye(size(obj.x,1)) - K*measmodel.H)*obj.P;   
        end
        
        function z_ingate = Gating(obj, z, measmodel)
            %GATING performs ellipsoidal gating
            %INPUT:  z: measurements --- (measurement dimension) x (number
            %of measurements) matrix
            %OUTPUT: z_ingate: measurements in the gate --- (measurement 
            %dimension) x (number of measurements in the gate) matrix
            zlength = size(z,2);
            in_gate = false(zlength,1);
            S = measmodel.H*obj.P*measmodel.H' + measmodel.R;
            nu = z - measmodel.H*repmat(obj.x,[1 zlength]);
            dist= sum((inv(chol(S))'*nu).^2);
            in_gate(dist<obj.gating.size) = true;
            z_ingate = z(:,in_gate);
        end
        
        function meas_likelihood = measLikelihood(obj, z, measmodel)
            %MEASLIKELIHOOD calculates the measurement update likelihood,
            %i.e., N(y;\hat{y},S).
            %INPUT:  z: measurements --- (measurement dimension) x (number
            %of measurements) matrix
            %OUTPUT: meas_likelihood: measurement update likelihood for
            %each measurement --- (measurement dimension) x 1 vector
            S = measmodel.H*obj.P*measmodel.H' + measmodel.R;
            meas_likelihood = mvnpdf(z',measmodel.H*obj.x,S);
        end
        
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
        
        function obj = nearestNeighborAssocLinearKalmanUpdate(obj, z, measmodel)
            %NEARESTNEIGHBORASSOCLINEARKALMANUPDATE performs nearest neighbor
            %data association and linear Kalman filter update
            %INPUT: z: measurements --- (measurement dimension) x (number
            %of measurements) matrix
            S = measmodel.H*obj.P*measmodel.H' + measmodel.R;
            nu = z - measmodel.H*repmat(obj.x,[1 size(z,2)]);
            
            %Perform ellipsoidal gating
            dist= sum((inv(chol(S))'*nu).^2);
            
            %Choose the closest measurement to the measurement prediction
            %in the gate. No measurement in the gate means missed detection
            %happens
            if min(dist) < obj.gating.size
                K = obj.P*measmodel.H'/S;
                obj.P = (eye(size(obj.x,1))-K*measmodel.H)*obj.P;
                [~,nn_idx] = min(dist);
                obj.x = obj.x + K*nu(:,nn_idx);
            end
        end
        
        function obj = probDataAssocLinearKalmanUpdate(obj, z, measmodel, sensormodel)
            %PROBDATAASSOCLINEARKALMANUPDATE performs probablistic data
            %association and linear kalman update
            %INPUT: z: measurements --- (measurement dimension) x (number
            %of measurements) matrix
            S = measmodel.H*obj.P*measmodel.H' + measmodel.R;
            nu = z - measmodel.H*repmat(obj.x,[1 size(z,2)]);
            
            %Perform ellipsoidal gating
            dist= sum((inv(chol(S))'*nu).^2);
            
            %No measurement in the gate means missed detection happens
            if min(dist) < obj.gating.size
                K = obj.P*measmodel.H'/S;
                s_d = size(obj.x,1);
                
                %Find all the measurements in the gate
                meas_update_idx = find(dist < obj.gating.size);
                num_meas_ingate = length(meas_update_idx);
                
                %Allocate memory
                mu = zeros(num_meas_ingate+1,1);
                x_temp = zeros(s_d,num_meas_ingate+1);
                P_temp = zeros(s_d,s_d,num_meas_ingate+1);
                
                %Missed detection
                mu(1) = (1-sensormodel.P_D*obj.gating.P_G)*(sensormodel.lambda_c)*(sensormodel.pdf_c);
                x_temp(:,1) = obj.x;
                P_temp(:,:,1) = obj.P;
                
                %For each measurment in the gate, perform Kalman update
                for i = 1:num_meas_ingate
                    mu(i+1) = mvnpdf(z(:,meas_update_idx(i)),measmodel.H*obj.x,S);
                    x_temp(:,i+1) = obj.x + K*nu(:,meas_update_idx(i));
                    P_temp(:,:,i+1) = (eye(size(obj.x,1))-K*measmodel.H)*obj.P;
                end
                
                %Normalise likelihoods
                mu = mu/sum(mu);
%                 %Calculate the equivalent measurement
%                 z_eq = mu(1)*measmodel.H*obj.x + z(:,meas_update_idx)*mu(2:end);
%                 obj.x = obj.x + K*(z_eq-measmodel.H*obj.x);

                %Calculate merged mean
                obj.x = x_temp*mu;
                %Calculate merged covariance
                for i = 1:num_meas_ingate+1
                    x_diff = x_temp(:,i)-obj.x;
                    obj.P = mu(i)*(P_temp(:,:,i) + x_diff*x_diff');
                end
            end
        end
        
        function [x_hat, P_hat] = GaussianMixtureReduction(w, x, P)
            %GAUSSIANMIXTUREREDUCTION: approximate a Gaussian mixture
            %density as a single Gaussian
            %INPUT: w: normalised weight of Gaussian components --- (number
            %of Gaussians) x 1 vector
            %       x: means of Gaussian components --- (variable dimension)
            %       x (number of Gaussians) matrix
            %       P: variances of Gaussian components --- (variable
            %       dimension) x (variable dimension) x (number of Gaussians) matrix
            %OUTPUT:x_hat: approximated mean --- (variable dimension) x 1
            %vector
            %       P_hat: approximated covariance --- (variable dimension)
            %       x (variable dimension) matrix
            x_hat = x*w;
            numGaussian = length(w);
            P_hat = zeros(size(P(:,:,1)));
            for i = 1:numGaussian
                x_diff = x(:,i) - x_hat;
                P_hat = w*(P(:,:,i) + x_diff*x_diff');
            end
        end
        
    end
end

