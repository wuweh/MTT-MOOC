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
            %         obj.hypothesis_reduction.wmin: allowed minimum hypothesis weight in logarithm domain --- scalar
            %         obj.hypothesis_reduction.merging_threshold: merging threshold --- scalar
            %         obj.hypothesis_reduction.M: allowed maximum number of hypotheses --- scalar
            obj.gating.P_G = P_G;
            obj.gating.size = chi2inv(obj.gating.P_G,m_d);
            obj.hypothesis_reduction.wmin = log(wmin);
            obj.hypothesis_reduction.merging_threshold = merging_threshold;
            obj.hypothesis_reduction.M = M;
            obj.x = x_0;
            obj.P = P_0;
        end
        
        function estimates = nearestNeighborTracker(obj, Z, motionmodel, measmodel)
            %NEARESTNEIGHBORTRACKER tracks a single target
            %using nearest neighbor association and kalman filter
            %INPUT: Z: cell array of size (total tracking time, 1), each cell 
            %          stores measurements of size (measurement dimension) x
            %          (number of measurements at corresponding time step)
            %OUTPUT:estimates: a structure with two fields:
            %               x: cell array of size (total tracking time, 1), each cell stores estimated target states
            %                  of size (target state dimension) x (number of targets
            %                  at corresponding time step)
            %               P: cell array of size (total tracking time, 1), each cell stores estimated target states
            %                  uncertainty covariance of size (target state dimension) x 
            %                  (target state dimension) x (number of targets
            %                  at corresponding time step)
            
            K = length(Z);
            estimates.x = cell(K,1);
            estimates.P = cell(K,1);
            for k = 1:K
                [obj.x, obj.P] = KalmanPredict(obj.x, obj.P, motionmodel);
                %Perform gating
                z_ingate = ellipsoidalGating(obj.x, obj.P, Z{k}, measmodel, obj.gating.size);
                if ~isempty(z_ingate)
                    meas_likelihood = measLikelihood(obj.x, obj.P, z_ingate, measmodel);
                    %Find the nearest neighbor in the gate
                    nearest_neighbor_meas = nearestNeighbor(z_ingate, meas_likelihood);
                    [obj.x, obj.P] = KalmanUpdate(obj.x, obj.P, nearest_neighbor_meas, measmodel);
                end
                estimates.x{k} = obj.x;
                estimates.P{k} = obj.P;
            end
        end
        
        function estimates = probDataAssocTracker(obj, Z, sensormodel, motionmodel, measmodel)
            %PROBDATAASSOCTRACKER tracks a single target
            %using probalistic data association and kalman filter
            %INPUT: Z: cell array of size (total tracking time, 1), each cell 
            %          stores measurements of size (measurement dimension) x
            %          (number of measurements at corresponding time step)
            %OUTPUT:estimates: a structure with two fields:
            %               x: cell array of size (total tracking time, 1), each cell stores estimated target states
            %                  of size (target state dimension) x (number of targets
            %                  at corresponding time step)
            %               P: cell array of size (total tracking time, 1), each cell stores estimated target states
            %                  uncertainty covariance of size (target state dimension) x 
            %                  (target state dimension) x (number of targets
            %                  at corresponding time step)
            K = length(Z);
            estimates.x = cell(K,1);
            estimates.P = cell(K,1);
            for k = 1:K
                %Prediction
                [obj.x, obj.P] = KalmanPredict(obj.x, obj.P, motionmodel);
                
                %Gating
                z_ingate = ellipsoidalGating(obj.x, obj.P, Z{k}, measmodel, obj.gating.size);
                
                if ~isempty(z_ingate)
                    %Allocate memory
                    num_meas_ingate = size(z_ingate,2);
                    mu = zeros(num_meas_ingate+1,1);
                    x_temp = zeros(motionmodel.d,num_meas_ingate+1);
                    P_temp = zeros(motionmodel.d,motionmodel.d,num_meas_ingate+1);
                    
                    %Missed detection hypothesis
                    [w_miss,x_temp(:,1),P_temp(:,:,1)] = ...
                        missDetectHypothesis(obj.x,obj.P,sensormodel.P_D,obj.gating.P_G);
                    mu(1) = w_miss+log(sensormodel.lambda_c)+log(sensormodel.pdf_c);
                    
                    %Measurement update hypothesis
                    [mu(2:end),x_temp(:,2:end),P_temp(:,:,2:end)] = ...
                        measUpdateHypothesis(obj.x,obj.P,z_ingate,measmodel,sensormodel.P_D);
                    
                    %Normalise likelihoods
                    [mu,~] = normalizeLogWeights(mu);
                    
                    %Merging
                    [obj.x,obj.P] = GaussianMixtureReduction(mu,x_temp,P_temp);
                end
                estimates.x{k} = obj.x;
                estimates.P{k} = obj.P;
            end
        end
        
        function estimates = multiHypothesesTracker(obj, Z, sensormodel, motionmodel, measmodel)
            %MULTIHYPOTHESESTRACKER tracks a single target using multiple
            %hypotheses
            %INPUT: Z: cell array of size (total tracking time, 1), each cell 
            %          stores measurements of size (measurement dimension) x
            %          (number of measurements at corresponding time step)
            %OUTPUT:estimates: a structure with two fields:
            %               x: cell array of size (total tracking time, 1), each cell stores estimated target states
            %                  of size (target state dimension) x (number of targets
            %                  at corresponding time step)
            %               P: cell array of size (total tracking time, 1), each cell stores estimated target states
            %                  uncertainty covariance of size (target state dimension) x 
            %                  (target state dimension) x (number of targets
            %                  at corresponding time step)
            
            %Initialize hypothesesWeight and multiHypotheses struct
            hypothesesWeight = 0;
            multiHypotheses = struct('x',obj.x,'P',obj.P);
            
            K = length(Z);
            estimates.x = cell(K,1);
            estimates.P = cell(K,1);
            for k = 1:K
                %Generate multiple new hypotheses
                [hypothesesWeight, multiHypotheses] = ...
                    multiHypothesesPropagation(hypothesesWeight, multiHypotheses, ...
                    Z{k}, obj.gating, sensormodel, motionmodel, measmodel);
                
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
                
                estimates.x{k} = obj.x;
                estimates.P{k} = obj.P;
            end
        end
        
    end
end

