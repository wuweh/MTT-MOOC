classdef singletargetracker
    %SINGLETARGETTRACKER is a class containing functions to track a single
    %target in clutter and missed detection. NO track initiation&deletion
    %logic.
    %Model structures need to be called:
    %sensormodel: a structure specifies the sensor parameters
    %           P_D: target detection probability --- scalar
    %           lambda_c: average number of clutter measurements
    %                   per time scan, Poisson distributed --- scalar
    %           pdf_c: clutter (Poisson) intensity --- scalar
    %motionmodel: a structure specifies the motion model parameters
    %           d: target state dimension --- scalar
    %           F: function handle return transition/Jacobian matrix
    %           f: function handle return predicted targe state
    %           Q: motion noise covariance matrix
    
    %measmodel: a structure specifies the measurement model parameters
    %           d: measurement dimension --- scalar
    %           H: function handle return transition/Jacobian matrix
    %           h: function handle return the observation of the target state
    %           R: measurement noise covariance matrix
    
    properties
        gating  %specify gating parameter
        x       %target state mean --- (target state dimension) x 1 vector
        P       %target state covariance --- (target state dimension) x (target state dimension) matrix
        hypothesis_reduction %specify hypothesis reduction parameter
    end
    
    methods
        
        function obj = initiator(obj,P_G,m_d,wmin,merging_threshold,M,x_0,P_0)
            %INITIATOR initiates singletargetracker class
            %INPUT: P_G: gating size in percentage --- scalar
            %       m_d: measurement dimension --- scalar
            %       wmin: allowed minimum hypothesis weight --- scalar
            %       merging_threshold: merging threshold --- scalar
            %       M: allowed maximum number of hypotheses --- scalar
            %       x_0: mean of target initial state --- (target state
            %           dimension) x 1 vector
            %       P_0: covariance of target initial state (target state
            %           dimension) x (target state dimension) matrix
            %OUTPUT:  obj.gating.P_G: gating size in percentage --- scalar
            %         obj.gating.size: gating size --- scalar
            %         obj.hypothesis_reduction.wmin: allowed minimum hypothesis weight --- scalar
            %         obj.hypothesis_reduction.merging_threshold: merging threshold --- scalar
            %         obj.hypothesis_reduction.M: allowed maximum number of hypotheses --- scalar
            obj.gating.P_G = P_G;
            obj.gating.size = chi2inv(obj.gating.P_G,m_d);
            obj.hypothesis_reduction.wmin = wmin;
            obj.hypothesis_reduction.merging_threshold = merging_threshold;
            obj.hypothesis_reduction.M = M;
            obj.x = x_0;
            obj.P = P_0;
        end
        
        function obj = nearestNeighborTracker(obj, z, motionmodel, measmodel)
            %NEARESTNEIGHBORTRACKER tracks a single target
            %using nearest neighbor association and kalman filter
            %INPUT: z: measurements --- (measurement dimension) x (number
            %           of measurements) matrix
            [obj.x, obj.P] = KalmanPredict(obj.x, obj.P, motionmodel);
            %Perform gating
            z_ingate = ellipsoidalGating(obj.x, obj.P, z, measmodel, obj.gating.size);
            if ~isempty(z_ingate)
                meas_likelihood = measLikelihood(obj.x, obj.P, z_ingate, measmodel);
                %Find the nearest neighbor in the gate
                nearest_neighbor_meas = nearestNeighbor(z_ingate, meas_likelihood);
                [obj.x, obj.P] = KalmanUpdate(obj.x, obj.P, nearest_neighbor_meas, measmodel);
            end
        end
        
        function obj = probDataAssocTracker(obj, z, sensormodel, motionmodel, measmodel)
            %PROBDATAASSOCTRACKER tracks a single target
            %using probalistic data association and kalman filter
            %INPUT: z: measurements --- (measurement dimension) x (number
            %       of measurements) matrix
            
            %Prediction
            [obj.x, obj.P] = KalmanPredict(obj.x, obj.P, motionmodel);
            
            %Gating
            z_ingate = ellipsoidalGating(obj.x, obj.P, z, measmodel, obj.gating.size);
            
            if ~isempty(z_ingate)
                %Allocate memory
                num_meas_ingate = size(z_ingate,2);
                mu = zeros(num_meas_ingate+1,1);
                x_temp = zeros(motionmodel.d,num_meas_ingate+1);
                P_temp = zeros(motionmodel.d,motionmodel.d,num_meas_ingate+1);
                
                %Missed detection hypothesis
                [w_miss,x_temp(:,1),P_temp(:,:,1)] = ...
                    missDetectHypothesis(obj.x,obj.P,sensormodel.P_D,obj.gating.P_G);
                mu(1) = w_miss*(sensormodel.lambda_c)*(sensormodel.pdf_c);
                
                %Measurement update hypothesis
                [mu(2:end),x_temp(:,2:end),P_temp(:,:,2:end)] = ...
                    measUpdateHypothesis(obj.x,obj.P,z_ingate,measmodel,sensormodel.P_D);
                
                %Normalise likelihoods
                mu = mu/sum(mu);
                
                %Merging
                [obj.x,obj.P] = GaussianMixtureReduction(mu,x_temp,P_temp);
            end
        end
        
        function [obj, hypothesesWeight, multiHypotheses] = multiHypothesesTracker(obj, ...
                hypothesesWeight, multiHypotheses, z, sensormodel, motionmodel, measmodel)
            %MULTIHYPOTHESESTRACKER tracks a single target using multiple
            %hypotheses
            %INPUT: hypothesesWeight: the weights of different hypotheses
            %                       --- (number of hypotheses) x 1 vector
            %       multiHypotheses: (number of hypotheses) x 1 structure
            %                       with two fields: x: target state mean;
            %                       P: target state covariance
            %       z: measurements --- (measurement dimension) x (number
            %           of measurements) matrix
            %OUTPUT:hypothesesWeight: the weights of different hypotheses
            %                       --- (number of updated hypotheses) x 1 vector
            %       multiHypotheses: (number of updated hypotheses) x 1 structure
            %                       with two fields: x: target state mean;
            %                       P: target state covariance
            
            %Generate multiple new hypotheses
            [hypothesesWeight, multiHypotheses] = ...
                multiHypothesesPropagation(hypothesesWeight, multiHypotheses, ...
                z, obj.gating, sensormodel, motionmodel, measmodel);
            
            %Prune hypotheses with weight smaller than the specified threshold
            [hypothesesWeight, multiHypotheses] = hypothesisReduction.prune(hypothesesWeight,...
                multiHypotheses, obj.hypothesis_reduction.wmin);
            
            %Keep at most M hypotheses with the highest weights
            [hypothesesWeight, multiHypotheses] = hypothesisReduction.cap(hypothesesWeight, ...
                multiHypotheses, obj.hypothesis_reduction.M);
            
            %Merge hypotheses within small enough Mahalanobis distance
            [hypothesesWeight,multiHypotheses] = hypothesisReduction.merge...
                (hypothesesWeight,multiHypotheses,obj.hypothesis_reduction.merging_threshold);
            
            %Extract target state from the hypothesis with the highest weight
            [~,idx] = max(hypothesesWeight);
            obj.x = multiHypotheses(idx).x;
            obj.P = multiHypotheses(idx).P;
        end
        
        function obj = nearestNeighborAssocLinearKalmanUpdate(obj, z, measmodel)
            %NEARESTNEIGHBORASSOCLINEARKALMANUPDATE performs nearest neighbor
            %data association and linear Kalman filter update
            %INPUT: z: measurements --- (measurement dimension) x (number
            %       of measurements) matrix
            S = measmodel.H*obj.P*measmodel.H' + measmodel.R;
            nu = z - measmodel.H*repmat(obj.x,[1 size(z,2)]);
            
            %Perform ellipsoidal gating
            dist = sum((inv(chol(S))'*nu).^2);
            
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
            %       of measurements) matrix
            S = measmodel.H*obj.P*measmodel.H' + measmodel.R;
            nu = z - measmodel.H*repmat(obj.x,[1 size(z,2)]);
            
            %Perform ellipsoidal gating
            dist = sum((inv(chol(S))'*nu).^2);
            
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
                    obj.P = mu(i).*(P_temp(:,:,i) + x_diff*x_diff');
                end
            end
        end
        
    end
end

