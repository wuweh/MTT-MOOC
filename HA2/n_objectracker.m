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
            %INPUT: state: structure array of size (1, number of objects) with two fields:
            %                x: object initial state mean --- (object state dimension) x 1 vector
            %                P: object initial state covariance --- (object state dimension) x (object state dimension) matrix
            %       Z: cell array of size (total tracking time, 1), each cell
            %          stores measurements of size (measurement dimension) x (number of measurements at corresponding time step)
            %OUTPUT:estimates: cell array of size (total tracking time, 1), each cell stores estimated object state of size (object state dimension) x (number of objects)
            
            
            n = length(states);
            K = length(Z);
            estimates = cell(K,1);
            
            for k = 1:K
                states = arrayfun(@(x) obj.density.predict(x, motionmodel), states);
                meas_in_gate_per_object = zeros(size(Z{k},2),n);
                for i = 1:n
                    [~,meas_in_gate_per_object(:,i)] = obj.density.ellipsoidalGating(states(i), Z{k}, measmodel, obj.gating.size);
                end
                
                used_meas_idx = sum(meas_in_gate_per_object,2) >= 1;
                meas_in_gate_per_object = logical(meas_in_gate_per_object(used_meas_idx,:));
                z_ingate = Z{k}(:,used_meas_idx);
                m = size(z_ingate,2);
                
                L1 = inf(n,m);
                for i = 1:n
                    L1(i,meas_in_gate_per_object(:,i)) = -obj.density.predictedLikelihood(states(i), z_ingate(:,meas_in_gate_per_object(:,i)), measmodel)'-log(sensormodel.P_D);
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
        
        function estimates = JPDAtracker(obj, states, Z, sensormodel, motionmodel, measmodel)
            %JPDATRACKER tracks n object using global nearest neighbor association
            %INPUT: obj: a class objective with properties:
            %            obj.density: density class handle with methods:
            %                        obj.density.expectedValue
            %                        obj.density.predict
            %                        obj.density.update
            %                        obj.density.predictedLikelihood
            %                        obj.density.ellipsoidalGating
            %                        obj.density.momentMatching
            %                        obj.density.mixtureReduction
            %            obj.gating.P_G: gating size in decimal --- scalar
            %            obj.gating.size: gating size --- scalar
            %            obj.hypothesis_reduction.wmin: allowed minimum hypothesis weight in logarithmic scale --- scalar
            %            obj.hypothesis_reduction.merging_threshold: merging threshold --- scalar
            %            obj.hypothesis_reduction.M: allowed maximum number of hypotheses --- scalar
            %       states: structure array of size (1, number of objects) with two fields:
            %                x: object initial state mean --- (object state dimension) x 1 vector
            %                P: object initial state covariance --- (object state dimension) x (object state dimension) matrix
            %       Z: cell array of size (total tracking time, 1), each cell stores measurements of size (measurement dimension) x (number of measurements at corresponding time step)
            %       sensormodel: a structure specifies the sensor parameters
            %           P_D: object detection probability --- scalar
            %           lambda_c: average number of clutter measurements per time scan, Poisson distributed --- scalar
            %           pdf_c: clutter (Poisson) intensity --- scalar
            %       motionmodel: a structure specifies the motion model parameters
            %           d: object state dimension --- scalar
            %           F: function handle return transition/Jacobian matrix
            %           f: function handle return predicted object state
            %           Q: motion noise covariance matrix
            %       measmodel: a structure specifies the measurement model parameters
            %           d: measurement dimension --- scalar
            %           H: function handle return transition/Jacobian matrix
            %           h: function handle return the observation of the target state
            %           R: measurement noise covariance matrix
            %OUTPUT:estimates: cell array of size (total tracking time, 1), each cell stores estimated object states in matrix form of size (object state dimension) x (number of objects)
            %DEPENDENCIES: singleobjecthypothesis.m
            %              GaussianDensity.m
            %              normalizeLogWeights.m
            %              hypothesisReduction.m
            %              kBest2DAssign.m
            
            n = length(states);
            K = length(Z);
            estimates = cell(K,1);
            
            for k = 1:K
                states = arrayfun(@(x) obj.density.predict(x, motionmodel), states);
                meas_in_gate_per_object = zeros(size(Z{k},2),n);
                for i = 1:n
                    [~,meas_in_gate_per_object(:,i)] = obj.density.ellipsoidalGating(states(i), Z{k}, measmodel, obj.gating.size);
                end
                
                used_meas_idx = sum(meas_in_gate_per_object,2) >= 1;
                meas_in_gate_per_object = logical(meas_in_gate_per_object(used_meas_idx,:));
                z_ingate = Z{k}(:,used_meas_idx);
                m = size(z_ingate,2);
                
                L1 = inf(n,m);
                for i = 1:n
                    L1(i,meas_in_gate_per_object(:,i)) = -obj.density.predictedLikelihood(states(i), z_ingate(:,meas_in_gate_per_object(:,i)), measmodel)'-log(sensormodel.P_D);
                end
                L2 = inf(n);
                L2(logical(eye(n))) = -(singleobjecthypothesis.undetected(sensormodel.P_D,obj.gating.P_G)+log(sensormodel.lambda_c)+log(sensormodel.pdf_c))*ones(n,1);
                L = [L1 L2];
                %Obtain M best assignments using Murty's algorithm
%                 [col4rowBest,~,gainBest]=kBest2DAssign(L,obj.hypothesis_reduction.M);
                %Obtain M low cost assignments using Gibbs sampling
                [col4rowBest,gainBest]= assign2DByGibbs(L,100,obj.hypothesis_reduction.M);
                
                %Normalize hypothesis weight
                normalizedWeight = normalizeLogWeights(-gainBest);
                
                %Prune hypotheses with weight smaller than the specified threshold
                hypo_idx = 1:length(normalizedWeight);
                [normalizedWeight, hypo_idx] = hypothesisReduction.prune(normalizedWeight,...
                    hypo_idx, obj.hypothesis_reduction.wmin);
                
                %Normalize hypothesis weight
                normalizedWeight = normalizeLogWeights(normalizedWeight);
                
                %Create new hypothesis according to the k-best assignments
                col4rowBest = col4rowBest(:,hypo_idx);
                numHypothesis = length(normalizedWeight);
                for i = 1:n
                    hypostates = repmat(states(i),[1,numHypothesis]);
                    for j = 1:numHypothesis
                        if col4rowBest(i,j) <= m
                            %Create object detection hypothesis
                            hypostates(j) = obj.density.update(states(i), z_ingate(:,col4rowBest(i,j)), measmodel);
                        end
                    end
                    %Merging
                    states(i) = obj.density.momentMatching(normalizedWeight, hypostates);
                    %Extract object state
                    estimates{k} = [estimates{k} obj.density.expectedValue(states(i))];
                end
            end
        end
        
        function estimates = TOMHT(obj, states, Z, sensormodel, motionmodel, measmodel)
            %TOMHT tracks n object using track-oriented multi-hypothesis tracking
            %INPUT: obj: a class objective with properties:
            %            obj.density: density class handle with methods:
            %                        obj.density.expectedValue
            %                        obj.density.predict
            %                        obj.density.update
            %                        obj.density.predictedLikelihood
            %                        obj.density.ellipsoidalGating
            %                        obj.density.momentMatching
            %                        obj.density.mixtureReduction
            %            obj.gating.P_G: gating size in decimal --- scalar
            %            obj.gating.size: gating size --- scalar
            %            obj.hypothesis_reduction.wmin: allowed minimum hypothesis weight in logarithmic scale --- scalar
            %            obj.hypothesis_reduction.merging_threshold: merging threshold --- scalar
            %            obj.hypothesis_reduction.M: allowed maximum number of hypotheses --- scalar
            %       states: structure array of size (1, number of objects) with two fields:
            %                x: object initial state mean --- (object state dimension) x 1 vector
            %                P: object initial state covariance --- (object state dimension) x (object state dimension) matrix
            %       Z: cell array of size (total tracking time, 1), each cell stores measurements of size (measurement dimension) x (number of measurements at corresponding time step)
            %       sensormodel: a structure specifies the sensor parameters
            %           P_D: object detection probability --- scalar
            %           lambda_c: average number of clutter measurements per time scan, Poisson distributed --- scalar
            %           pdf_c: clutter (Poisson) intensity --- scalar
            %       motionmodel: a structure specifies the motion model parameters
            %           d: object state dimension --- scalar
            %           F: function handle return transition/Jacobian matrix
            %           f: function handle return predicted object state
            %           Q: motion noise covariance matrix
            %       measmodel: a structure specifies the measurement model parameters
            %           d: measurement dimension --- scalar
            %           H: function handle return transition/Jacobian matrix
            %           h: function handle return the observation of the target state
            %           R: measurement noise covariance matrix
            %OUTPUT:estimates: cell array of size (total tracking time, 1), each cell stores estimated object states in matrix form of size (object state dimension) x (number of objects)
            %DEPENDENCIES: singleobjecthypothesis.m
            %              GaussianDensity.m
            %              normalizeLogWeights.m
            %              hypothesisReduction.m
            %              kBest2DAssign.m
            
            n = length(states);     %number of objects
            K = length(Z);          %total number of time steps
            estimates = cell(K,1);  %initialize estimates
            hypoTable = cell(n,1);  %initialize single object hypothesis table
            %each cell stores the single object hypotheses for the corresponding object
            for i = 1:n
                hypoTable{i} = states(i);
            end
            %initialize global hypotheses
            globalHypoWeight(1,1) = 0;
            globalHypo = ones(1,n);
            
            for k = 1:K
                m = size(Z{k},2);   %number of measurements at time step k
                %predict for each single object hypothesis for each object
                hypoTable = cellfun(@(y) arrayfun(@(x) obj.density.predict(x, motionmodel), y),hypoTable,'UniformOutput',false);
                
                %initialize likelihood table and updated single object hypothesis table
                likTable = cell(n,1);
                hypoTableUpd = cell(n,1);
                for i = 1:n
                    %number of single object hypotheses for object i
                    num_hypo_per_object = length(hypoTable{i});
                    %initialize gating table, recording whether measurement
                    %m is inside the gate of single object hypothesis h
                    meas_in_gate_per_object = zeros(m,num_hypo_per_object);
                    likTable{i} = inf(num_hypo_per_object,m+1);
                    
                    hypoTableUpd{i} = cell(num_hypo_per_object*(m+1),1);
                    for j = 1:num_hypo_per_object
                        %ellipsoidal gating
                        [~,meas_in_gate_per_object(:,j)] = obj.density.ellipsoidalGating(hypoTable{i}(j), Z{k}, measmodel, obj.gating.size);
                        %missed detection likelihood
                        likTable{i}(j,1) = -(singleobjecthypothesis.undetected...
                            (sensormodel.P_D,obj.gating.P_G)+log(sensormodel.lambda_c)+log(sensormodel.pdf_c));
                        %predicted likelihood
                        likTable{i}(j,[false;logical(meas_in_gate_per_object(:,j))]) ...
                            = -obj.density.predictedLikelihood(hypoTable{i}(j), Z{k}(:,logical(meas_in_gate_per_object(:,j))), measmodel)'-log(sensormodel.P_D);
                        %update step, only consider measurements inside the gate
                        hypoTableUpd{i}{(j-1)*(m+1)+1} = hypoTable{i}(j);
                        for jj = 1:m
                            if meas_in_gate_per_object(jj,j) == 1
                                hypoTableUpd{i}{(j-1)*(m+1)+jj+1} = obj.density.update(hypoTable{i}(j), Z{k}(:,jj), measmodel);
                            end
                        end
                    end
                end
                
                globalHypoWeightUpd = [];
                globalHypoUpd = zeros(0,n);
                num_new_hypo = 0;
                H = size(globalHypo,1);
                for h = 1:H
                    %Create assignment matrix
                    L1 = inf(n,m);
                    L2 = inf(n);
                    for i = 1:n
                        L1(i,:) = likTable{i}(globalHypo(h,i),2:end);
                        L2(i,i) = likTable{i}(globalHypo(h,i),1);
                    end
                    L = [L1 L2];
                    %Obtain M best assignments using Murty's algorithm
%                     [col4rowBest,~,gainBest]=kBest2DAssign(L,ceil(exp(globalHypoWeight(h))*obj.hypothesis_reduction.M));
                    %Obtain M low cost assignments using Gibbs sampling
                    [col4rowBest,gainBest]= assign2DByGibbs(L,100,ceil(exp(globalHypoWeight(h))*obj.hypothesis_reduction.M));
                    
                    col4rowBest(col4rowBest>m) = 0;
                    globalHypoWeightUpd = [globalHypoWeightUpd;-gainBest+globalHypoWeight(h)];
                    %Update look-up table
                    for j = 1:length(gainBest)
                        num_new_hypo = num_new_hypo + 1;
                        for i = 1:n
                            globalHypoUpd(num_new_hypo,i) = (globalHypo(h,i)-1)*(m+1) + col4rowBest(i,j) + 1;
                        end
                    end
                end
                
                %Normalize global hypothesis weights
                globalHypoWeight = normalizeLogWeights(globalHypoWeightUpd);
                
                %Prune hypotheses with weight smaller than the specified threshold
                hypo_idx = 1:num_new_hypo;
                [globalHypoWeight, hypo_idx] = hypothesisReduction.prune(globalHypoWeight,...
                    hypo_idx, obj.hypothesis_reduction.wmin);
                globalHypoUpd = globalHypoUpd(hypo_idx,:);
                globalHypoWeight = normalizeLogWeights(globalHypoWeight);
                
                %Keep at most M hypotheses with the highest weights
                hypo_idx = 1:length(globalHypoWeight);
                [globalHypoWeight, hypo_idx] = hypothesisReduction.cap(globalHypoWeight, ...
                    hypo_idx, obj.hypothesis_reduction.M);
                globalHypo = globalHypoUpd(hypo_idx,:);
                globalHypoWeight = normalizeLogWeights(globalHypoWeight);
                
                %Prune local hypotheses that not appear in maintained global hypotheses
                for i = 1:n
                    hypoTableTemp = hypoTableUpd{i}(unique(globalHypo(:,i)));
                    hypoTable{i} = [hypoTableTemp{:}];
                end
                
                %Clean hypothesis table
                for i = 1:n
                    [~,~,globalHypo(:,i)] = unique(globalHypo(:,i),'rows');
                end
                
                %Extract object states
                [~,I] = max(globalHypoWeight);
                bestHypo = globalHypo(I,:);
                for i = 1:n
                    estimates{k} = [estimates{k} obj.density.expectedValue(hypoTable{i}(bestHypo(i)))];
                end
                
            end
        end
        
    end
end

