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
            %UPDATE
            w_upd = obj.paras.w + singleobjecthypothesis.undetected(sensormodel.P_D,gating.P_G);
            states_upd = obj.paras.states;
            %DETECTED
            num_component = length(obj.paras.w);
            for i = 1:num_component
                z_ingate = obj.density.ellipsoidalGating(obj.paras.states(i), z, measmodel, gating.size);
                if ~isempty(z_ingate)
                    [states_detected,l_detected] = singleobjecthypothesis.detected(obj.density,obj.paras.states(i),z_ingate,measmodel,sensormodel.P_D);

                    w_temp = [obj.paras.w(i)+l_detected;log(sensormodel.lambda_c)+log(sensormodel.pdf_c)];
                    w_temp = normalizeLogWeights(w_temp);

                    w_upd = [w_upd;w_temp(1:end-1)];
                    states_upd = [states_upd;states_detected];
                end
            end
            obj.paras.w = w_upd;
            obj.paras.states = states_upd;
        end
        
        function obj = componentReduction(obj,hypothesis_reduction)
            %PRUNE
            [~,log_sum_w] = normalizeLogWeights(obj.paras.w);
            [obj.paras.w, obj.paras.states] = hypothesisReduction.prune(obj.paras.w, obj.paras.states, hypothesis_reduction.wmin);
            obj.paras.w = obj.paras.w + log_sum_w;
            %MERGING
            [~,log_sum_w] = normalizeLogWeights(obj.paras.w);
            [obj.paras.w, obj.paras.states] = hypothesisReduction.merge(obj.paras.w, obj.paras.states, hypothesis_reduction.merging_threshold, obj.density);
            obj.paras.w = obj.paras.w + log_sum_w;
            %CAPPING
            [~,log_sum_w] = normalizeLogWeights(obj.paras.w);
            [obj.paras.w, obj.paras.states] = hypothesisReduction.cap(obj.paras.w, obj.paras.states, hypothesis_reduction.M);
            obj.paras.w = obj.paras.w + log_sum_w;
        end
        
        function estimates = stateExtraction(obj)
            estimates = [];
            idx = find(obj.paras.w > log(0.5));
            if ~isempty(idx)
                for j = 1:length(idx)
                    repeat_num_targets = round(exp(obj.paras.w(idx(j))));
                    state = obj.density.expectedValue(obj.paras.states(idx(j)));
                    estimates = [estimates repmat(state,[1,repeat_num_targets])];
                end
            end
        end
        
    end
    
end

