classdef PHDfilter
    %PHDFILTER is a class containing necessary functions to implement the PHD filter
    %DEPENDENCIES: singleobjecthypothesis.m
    %              GaussianDensity.m
    %              normalizeLogWeights.m
    %              hypothesisReduction.m
    
    properties
        density %density class handle
        paras   %parameters specify a PPP
    end
    
    methods
        function obj = initialize(obj,density_class_handle,birthmodel)
            %INITIATOR initializes PHDfilter class
            %INPUT: density_class_handle: density class handle
            %       birthmodel: a struct specifying the intensity (mixture) of a PPP birth model
            %OUTPUT:obj.density: density class handle
            %       obj.paras.w: weights of mixture components --- vector
            %                    of size (number of mixture components x 1)
            %       obj.paras.states: parameters of mixture components ---
            %                    struct array of size (number of mixture components x 1)
            obj.density = density_class_handle;
            obj.paras.w = log([birthmodel.w]');
            obj.paras.states = rmfield(birthmodel,'w')';
        end
        
        function obj = predict(obj,motionmodel,P_S,birthmodel)
            %PREDICT performs PPP prediction step
            %INPUT: P_S: object survival probability
            %       birthmodel: a struct specifying the intensity (mixture) of a PPP birth model
            obj.paras.w = obj.paras.w + log(P_S);
            obj.paras.states = arrayfun(@(x) obj.density.predict(x, motionmodel), obj.paras.states);
            %Incorporate birth terms into predicted PPP
            obj.paras.w = [obj.paras.w;[birthmodel.w]'];
            obj.paras.states = [obj.paras.states;rmfield(birthmodel,'w')'];
        end
        
        function obj = update(obj,z,measmodel,sensormodel,gating)
            %UPDATE performs PPP update step and PPP approximation
            %INPUT: gating: a struct with two fields: P_G, size used to
            %               specify the gating parameters
            
            %update weights of mixture compoenent resulted from missed detection
            w_upd = obj.paras.w + singleobjecthypothesis.undetected(sensormodel.P_D,gating.P_G);
            states_upd = obj.paras.states;
            
            n = length(obj.paras.w);
            %perform gating for each mixture component
            meas_in_gate_per_object = zeros(size(z,2),n);
            for i = 1:n
                [~,meas_in_gate_per_object(:,i)] = obj.density.ellipsoidalGating(obj.paras.states(i),z,measmodel,gating.size);
            end
            used_meas_idx = sum(meas_in_gate_per_object,2) >= 1;
            %returns a matrix with boolean element (j,i) specifying whether
            %measurement j falls inside the gate formed by component i
            meas_in_gate_per_object = logical(meas_in_gate_per_object(used_meas_idx,:));
            z_ingate = z(:,used_meas_idx);
            
            m = size(z_ingate,2);
            w = zeros(size(meas_in_gate_per_object));
            %update weights of mixture compoenent resulted from measurement update
            for i = 1:n
                if any(meas_in_gate_per_object(:,i))
                    [states_i,w(meas_in_gate_per_object(:,i),i)] = ...
                        singleobjecthypothesis.detected(obj.density,obj.paras.states(i),z_ingate(:,meas_in_gate_per_object(:,i)),measmodel,sensormodel.P_D);
                    states_upd = [states_upd;states_i];
                    w(meas_in_gate_per_object(:,i),i) = w(meas_in_gate_per_object(:,i),i) + obj.paras.w(i);
                end
            end
            %normalise weights of mixture components resulted from being
            %updated by the same measurement. Without PPP approximation, w
            %resulted from measurement update can be regarded as the
            %probability of existence of Bernoulli components.
            for j = 1:m
                w_temp = [w(j,meas_in_gate_per_object(j,:)) log(sensormodel.lambda_c)+log(sensormodel.pdf_c)];
                w_temp = normalizeLogWeights(w_temp);
                w(j,meas_in_gate_per_object(j,:)) = w_temp(1:end-1);
            end
            for i = 1:n
                w_upd = [w_upd;w(meas_in_gate_per_object(:,i),i)];
            end
            
            %Alternative implementation
            %             for j = 1:m
            %                 num_objects_in_gate = sum(meas_in_gate_per_object(j,:));
            %                 states_idx = find(meas_in_gate_per_object(j,:)==1);
            %                 states_j = repmat(states_upd(1),[num_objects_in_gate,1]);
            %                 w_j = zeros(num_objects_in_gate,1);
            %                 for i = 1:num_objects_in_gate
            %                     [states_j(i),w_j(i)] = singleobjecthypothesis.detected...
            %                           (obj.density,obj.paras.states(states_idx(i)),z_ingate(:,j),measmodel,sensormodel.P_D);
            %                     w_j(i) = w_j(i) + obj.paras.w(states_idx(i));
            %                 end
            %
            %                 w_temp = [w_j;log(sensormodel.lambda_c)+log(sensormodel.pdf_c)];
            %                 w_temp = normalizeLogWeights(w_temp);
            %
            %                 w_upd = [w_upd;w_temp(1:end-1)];
            %                 states_upd = [states_upd;states_j];
            %             end
            
            obj.paras.w = w_upd;
            obj.paras.states = states_upd;
        end
        
        function obj = componentReduction(obj,hypothesis_reduction)
            %COMPONENTREDUCTION approximates the PPP by representing its
            %intensity with fewer parameters
            %pruning
            [obj.paras.w, obj.paras.states] = hypothesisReduction.prune(obj.paras.w, obj.paras.states, hypothesis_reduction.wmin);
            %merging
            [obj.paras.w, obj.paras.states] = hypothesisReduction.merge(obj.paras.w, obj.paras.states, hypothesis_reduction.merging_threshold, obj.density);
            %capping
            [obj.paras.w, obj.paras.states] = hypothesisReduction.cap(obj.paras.w, obj.paras.states, hypothesis_reduction.M);
        end
        
        function estimates = PHD_estimator(obj,threshold)
            %PHD_ESTIMATOR performs object state estimation in the GMPHD filter
            %INPUT: threshold (if exist): object states are extracted from
            %       Gaussian components with weight no less than this threhold in Estimator 1.
            %OUTPUT:estimates: estimated object states in matrix form of
            %                  size (object state dimension) x (number of objects)
            %%%
            %In Estimator 1, we select the means of the Gaussians that have 
            %weights greater than some threshold.
            %In Estimator 2, we first obtain the estimated cardinality mean
            %by summing up the weights of Gaussian components. The number of 
            %objects is determined by the rounded nearest integer. The
            %estimation is accurate when the object detection probability
            %is high. 
            estimates = [];
            if nargin == 2
                %Estimator 1
                %implementation proposed in Ba-Vo's GM-PHD paper
                idx = find(obj.paras.w > log(threshold));
                if ~isempty(idx)
                    for j = 1:length(idx)
                        %closely spaced Gaussian components might have been
                        %merged, if w is too large, duplicate the state
                        repeat_num_targets= round(exp(obj.paras.w(idx(j))));
                        state = repmat(obj.density.expectedValue(obj.paras.states(idx(j))),[1,repeat_num_targets]);
                        estimates = [estimates state];
                    end
                end
            else
                %Estimator 2
                %obtain the estimated cardinality mean. The estimation error 
                %is small if P_D is close to one.
                n = round(sum(exp(obj.paras.w)));
                %extract object states from the n components with the highest weights
                if n > 0
                    [~,I] = sort(obj.paras.w,'descend');
                    for j = 1:n
                        state = obj.density.expectedValue(obj.paras.states(I(j)));
                        estimates = [estimates state];
                    end
                end
            end
        end
        
    end
    
end

