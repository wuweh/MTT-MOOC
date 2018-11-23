classdef singleobjectracker
    %SINGLEOBJECTRACKER is a class containing functions to track a single object in clutter.
    %Model structures need to be called:
    %sensormodel: a structure specifies the sensor parameters
    %           P_D: object detection probability --- scalar
    %           lambda_c: average number of clutter measurements per time scan, Poisson distributed --- scalar
    %           pdf_c: clutter (Poisson) intensity --- scalar
    %motionmodel: a structure specifies the motion model parameters
    %           d: object state dimension --- scalar
    %           F: function handle return transition/Jacobian matrix
    %           f: function handle return predicted object state
    %           Q: motion noise covariance matrix
    %measmodel: a structure specifies the measurement model parameters
    %           d: measurement dimension --- scalar
    %           H: function handle return transition/Jacobian matrix
    %           h: function handle return the observation of the target state
    %           R: measurement noise covariance matrix
    
    properties
        gating  %specify gating parameter
        hypothesis_reduction %specify hypothesis reduction parameter
        density %density class handle
    end
    
    methods
        
        function obj = initialize(obj,density_class_handle,P_G,m_d,wmin,merging_threshold,M)
            %INITIATOR initializes singleobjectracker class
            %INPUT: density_class_handle: density class handle
            %       P_G: gating size in percentage --- scalar
            %       m_d: measurement dimension --- scalar
            %       wmin: allowed minimum hypothesis weight --- scalar
            %       merging_threshold: merging threshold --- scalar
            %       M: allowed maximum number of hypotheses --- scalar
            %OUTPUT:  obj.density: density class handle
            %         obj.gating.P_G: gating size in percentage --- scalar
            %         obj.gating.size: gating size --- scalar
            %         obj.hypothesis_reduction.wmin: allowed minimum hypothesis weight in logarithm domain --- scalar
            %         obj.hypothesis_reduction.merging_threshold: merging threshold --- scalar
            %         obj.hypothesis_reduction.M: allowed maximum number of hypotheses --- scalar
            obj.density = feval(density_class_handle);
            obj.gating.P_G = P_G;
            obj.gating.size = chi2inv(obj.gating.P_G,m_d);
            obj.hypothesis_reduction.wmin = log(wmin);
            obj.hypothesis_reduction.merging_threshold = merging_threshold;
            obj.hypothesis_reduction.M = M;
        end
        
        function estimates = nearestNeighborTracker(obj, state, Z, motionmodel, measmodel)
            %NEARESTNEIGHBORTRACKER tracks a single object using nearest neighbor association
            %INPUT: state: a structure with two fields:
            %                x: object initial state mean --- (object state dimension) x 1 vector
            %                P: object initial state covariance --- (object state dimension) x (object state dimension) matrix
            %       Z: cell array of size (total tracking time, 1), each cell stores measurements of 
            %            size (measurement dimension) x (number of measurements at corresponding time step)
            %OUTPUT:estimates: cell array of size (total tracking time, 1), each cell stores estimated object state of size (object state dimension) x 1
            
            K = length(Z);
            estimates = cell(K,1);
            for k = 1:K
                state = obj.density.predict(state, motionmodel);
                %Perform gating
                z_ingate = obj.density.ellipsoidalGating(state, Z{k}, measmodel, obj.gating.size);
                if ~isempty(z_ingate)
                    predict_likelihood = obj.density.predictedLikelihood(state, z_ingate, measmodel);
                    %Find the nearest neighbor in the gate
                    [~, nearest_neighbor_assoc] = max(predict_likelihood);
                    nearest_neighbor_meas = z_ingate(:,nearest_neighbor_assoc);
                    state = obj.density.update(state, nearest_neighbor_meas, measmodel);
                end
                estimates{k} = obj.density.expectedValue(state);
            end
        end
        
        
        function estimates = probDataAssocTracker(obj, state, Z, sensormodel, motionmodel, measmodel)
            %PROBDATAASSOCTRACKER tracks a single object using probalistic data association
            %INPUT: state: a structure with two fields:
            %                x: object initial state mean --- (object state dimension) x 1 vector
            %                P: object initial state covariance --- (object state dimension) x (object state dimension) matrix
            %       Z: cell array of size (total tracking time, 1), each cell
            %          stores measurements of size (measurement dimension) x (number of measurements at corresponding time step)
            %OUTPUT:estimates: cell array of size (total tracking time, 1), each cell stores estimated object state of size (object state dimension) x 1
            
            K = length(Z);
            estimates = cell(K,1);
            for k = 1:K
                %Prediction
                state = obj.density.predict(state, motionmodel);
                
                %Gating
                z_ingate = obj.density.ellipsoidalGating(state, Z{k}, measmodel, obj.gating.size);
                
                if ~isempty(z_ingate)
                    %Allocate memory
                    num_meas_ingate = size(z_ingate,2);
                    mu = zeros(num_meas_ingate+1,1);
                    
                    %Missed detection hypothesis
                    hypothesized_state(1,1) = state;
                    w_miss = singleobjecthypothesis.undetected(sensormodel.P_D,obj.gating.P_G);
                    mu(1) = w_miss+log(sensormodel.lambda_c)+log(sensormodel.pdf_c);
                    
                    %Measurement update hypothesis
                    [hypothesized_state(2:num_meas_ingate+1,1),mu(2:num_meas_ingate+1)] = ...
                        singleobjecthypothesis.detected(obj.density,state,z_ingate,measmodel,sensormodel.P_D);
                    
                    %Normalise likelihoods
                    [mu,~] = normalizeLogWeights(mu);
                    
                    %Merging
                    state = obj.density.momentMatching(mu, hypothesized_state);
                    %Free memory
                    clear hypothesized_state;
                end
                estimates{k} = obj.density.expectedValue(state);
            end
        end
        
        function estimates = multiHypothesesTracker(obj, state, Z, sensormodel, motionmodel, measmodel)
            %MULTIHYPOTHESESTRACKER tracks a single object using multiple hypotheses solution
            %INPUT: state: a structure with two fields:
            %                x: object initial state mean --- (object state dimension) x 1 vector
            %                P: object initial state covariance --- (object state dimension) x (object state dimension) matrix
            %       Z: cell array of size (total tracking time, 1), each cell
            %          stores measurements of size (measurement dimension) x (number of measurements at corresponding time step)
            %OUTPUT:estimates: cell array of size (total tracking time, 1), each cell stores estimated object state of size (object state dimension) x 1
            
            %Initialize hypothesesWeight and multiHypotheses struct
            hypothesesWeight(1,1) = 0;
            multiHypotheses(1,1) = state;
            
            K = length(Z);
            estimates = cell(K,1);
            for k = 1:K
                num_hypotheses = length(multiHypotheses);
                
                %Prediction
                for i = 1:num_hypotheses
                    multiHypotheses(i) = obj.density.predict(multiHypotheses(i), motionmodel);
                end
                
                hypothesesWeightUpdate = zeros(num_hypotheses,1);
                
                %For each hypothesis, generate missed detection hypotheses
                for i = 1:num_hypotheses
                    multiHypothesesUpdate(i,1) = multiHypotheses(i);
                    w_miss = singleobjecthypothesis.undetected(sensormodel.P_D, obj.gating.P_G);
                    hypothesesWeightUpdate(i,1) = hypothesesWeight(i)+w_miss+log(sensormodel.lambda_c)+log(sensormodel.pdf_c);
                end
                
                %For each hypothesis, generate measurement update hypotheses
                idx = num_hypotheses;
                for i = 1:num_hypotheses
                    %Perform gating for each hypothesis, only generate hypotheses for
                    %measurements in the gate
                    z_ingate = obj.density.ellipsoidalGating(multiHypotheses(i), Z{k}, measmodel, obj.gating.size);
                    if ~isempty(z_ingate)
                        num_meas_ingate = size(z_ingate,2);
                        [multiHypothesesUpdate(idx+1:idx+num_meas_ingate,1),wupd] = singleobjecthypothesis.detected(obj.density,multiHypotheses(i),z_ingate,measmodel,sensormodel.P_D);
                        hypothesesWeightUpdate(idx+1:idx+num_meas_ingate,1) = wupd+hypothesesWeight(i);
                        idx = idx + num_meas_ingate;
                    end
                end

                multiHypotheses = multiHypothesesUpdate;
                hypothesesWeight = hypothesesWeightUpdate;
                
                %Normalize hypotheses weights
                [hypothesesWeight,~] = normalizeLogWeights(hypothesesWeight);
                
                %Prune hypotheses with weight smaller than the specified threshold
                [hypothesesWeight, multiHypotheses] = hypothesisReduction.prune(hypothesesWeight,...
                    multiHypotheses, obj.hypothesis_reduction.wmin);
                
                %Keep at most M hypotheses with the highest weights
                [hypothesesWeight, multiHypotheses] = hypothesisReduction.cap(hypothesesWeight, ...
                    multiHypotheses, obj.hypothesis_reduction.M);
                
                %Merge hypotheses within small enough Mahalanobis distance
                [hypothesesWeight,multiHypotheses] = hypothesisReduction.merge...
                    (hypothesesWeight,multiHypotheses,obj.hypothesis_reduction.merging_threshold,obj.density);
                
                %Extract target state from the hypothesis with the highest weight
                [~,idx] = max(hypothesesWeight);
                estimates{k} = obj.density.expectedValue(multiHypotheses(idx));
            end
        end
        
        
    end
end

