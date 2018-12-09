classdef n_objectracker
    %N_OBJECTRACKER is a class containing functions to track n object in clutter.
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
            %       P_G: gating size in decimal --- scalar
            %       m_d: measurement dimension --- scalar
            %       wmin: allowed minimum hypothesis weight --- scalar
            %       merging_threshold: merging threshold --- scalar
            %       M: allowed maximum number of hypotheses --- scalar
            %OUTPUT:  obj.density: density class handle
            %         obj.gating.P_G: gating size in decimal --- scalar
            %         obj.gating.size: gating size --- scalar
            %         obj.hypothesis_reduction.wmin: allowed minimum hypothesis weight in logarithmic scale --- scalar
            %         obj.hypothesis_reduction.merging_threshold: merging threshold --- scalar
            %         obj.hypothesis_reduction.M: allowed maximum number of hypotheses --- scalar
            obj.density = feval(density_class_handle);
            obj.gating.P_G = P_G;
            obj.gating.size = chi2inv(obj.gating.P_G,m_d);
            obj.hypothesis_reduction.wmin = log(wmin);
            obj.hypothesis_reduction.merging_threshold = merging_threshold;
            obj.hypothesis_reduction.M = M;
        end
        
        function estimates = GNNtracker(obj, states, Z, sensormodel, motionmodel, measmodel)
            %GNNTRACKER tracks n object using global nearest neighbor association
            
            n = length(states);
            K = length(Z);
            estimates = cell(K,1);
            
            for k = 1:K
                meas_in_gate_per_object = zeros(size(Z{k},2),n);
                states = arrayfun(@(x) obj.density.predict(x, motionmodel), states);
                for i = 1:n
                    [~,meas_in_gate_per_object(:,i)] = obj.density.ellipsoidalGating(states(i), Z{k}, measmodel, obj.gating.size);
                end
                
                used_meas_idx = sum(meas_in_gate_per_object,2) >= 1;
                meas_in_gate_per_object = logical(meas_in_gate_per_object(used_meas_idx,:));
                z_ingate = Z{k}(:,used_meas_idx);
                m = size(z_ingate,2);
                
                L1 = inf(n,m);
                for i = 1:n
                    L1(i,meas_in_gate_per_object(:,i)) = -obj.density.predictedLikelihood(states(i), z_ingate(:,meas_in_gate_per_object(:,i)), measmodel)';
                end
                L2 = inf(n);
                L2(logical(eye(n))) = -(singleobjecthypothesis.undetected(sensormodel.P_D,obj.gating.P_G)+log(sensormodel.lambda_c)+log(sensormodel.pdf_c))*ones(n,1);
                col4row = assign2D([L1 L2]);
                
                for i = 1:n
                    if col4row(i) <= m
                        states(i) = obj.density.update(states(i), z_ingate(:,col4row(i)), measmodel);
                    end
                    estimates{k} = [estimates{k} obj.density.expectedValue(states(i))];
                end
            end
        end
        
        function estimates = TOMHT(obj, states, Z, sensormodel, motionmodel, measmodel)
            %TOMHT tracks n object using track-oriented multi-hypothesis tracking
            
            n = length(states);
            K = length(Z);
            estimates = cell(K,1);
            
            for k = 1:K
                meas_in_gate_per_object = zeros(size(Z{k},2),n);
                states = arrayfun(@(x) obj.density.predict(x, motionmodel), states);
                for i = 1:n
                    [~,meas_in_gate_per_object(:,i)] = obj.density.ellipsoidalGating(states(i), Z{k}, measmodel, obj.gating.size);
                end
                
                used_meas_idx = sum(meas_in_gate_per_object,2) >= 1;
                meas_in_gate_per_object = logical(meas_in_gate_per_object(used_meas_idx,:));
                z_ingate = Z{k}(:,used_meas_idx);
                m = size(z_ingate,2);
                
                L1 = inf(n,m);
                for i = 1:n
                    L1(i,meas_in_gate_per_object(:,i)) = -obj.density.predictedLikelihood(states(i), z_ingate(:,meas_in_gate_per_object(:,i)), measmodel)';
                end
                L2 = inf(n);
                L2(logical(eye(n))) = -singleobjecthypothesis.undetected(sensormodel.P_D,obj.gating.P_G)*ones(n,1);
                L = [L1 L2];
                col4row = assign2D(L);
                
                for i = 1:n
                    if col4row(i) <= m
                        states(i) = obj.density.update(states(i), z_ingate(:,col4row(i)), measmodel);
                    end
                    estimates{k} = [estimates{k} obj.density.expectedValue(states(i))];
                end
            end
        end
        
    end
end

