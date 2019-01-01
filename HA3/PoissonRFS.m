classdef PoissonRFS
    %POISSONRFS
    
    properties
        density
        paras
    end
    
    methods
        function obj = initialize(obj,density_class_handle,birthmodel)
            %INITIATOR
            obj.density = density_class_handle;
            obj.paras.w = log([birthmodel.w]');
            obj.paras.states = rmfield(birthmodel,'w')';
        end
        
        function obj = predict(obj,motionmodel,P_S,birthmodel)
            %PREDICT
            obj.paras.w = obj.paras.w + log(P_S);
            obj.paras.states = arrayfun(@(x) obj.density.predict(x, motionmodel), obj.paras.states);
            %Incorporate birth terms
            obj.paras.w = [obj.paras.w;[birthmodel.w]'];
            obj.paras.states = [obj.paras.states;rmfield(birthmodel,'w')'];
        end
        
        function obj = update(obj,z,measmodel,sensormodel,gating)
            %Undetected objects
            w_upd = obj.paras.w + singleobjecthypothesis.undetected(sensormodel.P_D,gating.P_G);
            states_upd = obj.paras.states;
            
            %Detected objects
            n = length(obj.paras.w);
            meas_in_gate_per_object = zeros(size(z,2),n);
            for i = 1:n
                [~,meas_in_gate_per_object(:,i)] = obj.density.ellipsoidalGating(obj.paras.states(i),z,measmodel,gating.size);
            end
            used_meas_idx = sum(meas_in_gate_per_object,2) >= 1;
            meas_in_gate_per_object = logical(meas_in_gate_per_object(used_meas_idx,:));
            z_ingate = z(:,used_meas_idx);
            m = size(z_ingate,2);
            for j = 1:m
                num_objects_in_gate = sum(meas_in_gate_per_object(j,:));
                states_idx = find(meas_in_gate_per_object(j,:)==1);
                states_j = repmat(states_upd(1),[num_objects_in_gate,1]);
                w_j = zeros(num_objects_in_gate,1);
                for i = 1:num_objects_in_gate
                    [states_j(i),w_j(i)] = singleobjecthypothesis.detected(obj.density,obj.paras.states(states_idx(i)),z_ingate(:,j),measmodel,sensormodel.P_D);
                    w_j(i) = w_j(i) + obj.paras.w(states_idx(i));
                end
                
                w_temp = [w_j;log(sensormodel.lambda_c)+log(sensormodel.pdf_c)];
                w_temp = normalizeLogWeights(w_temp);
                
                w_upd = [w_upd;w_temp(1:end-1)];
                states_upd = [states_upd;states_j];
            end
            
            obj.paras.w = w_upd;
            obj.paras.states = states_upd;
        end
        
        function obj = componentReduction(obj,hypothesis_reduction)
            %PRUNE
            [obj.paras.w, obj.paras.states] = hypothesisReduction.prune(obj.paras.w, obj.paras.states, hypothesis_reduction.wmin);
            %MERGING
            [obj.paras.w, obj.paras.states] = hypothesisReduction.merge(obj.paras.w, obj.paras.states, hypothesis_reduction.merging_threshold, obj.density);
            %CAPPING
            [obj.paras.w, obj.paras.states] = hypothesisReduction.cap(obj.paras.w, obj.paras.states, hypothesis_reduction.M);
        end
        
        function estimates = stateExtraction(obj)
            estimates = [];
            idx = find(obj.paras.w > log(0.5));
            if ~isempty(idx)
                for j = 1:length(idx)
                    state = obj.density.expectedValue(obj.paras.states(idx(j)));
                    estimates = [estimates state];
                end
            end
        end
        
    end
    
end

