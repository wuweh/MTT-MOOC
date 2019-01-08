classdef PMBMfilter
    
    properties
        density %density class handle
        paras
    end
    
    methods
        function obj = initialize(obj,density_class_handle,birthmodel)
            obj.density = density_class_handle;
            obj.paras.PPP.w = log([birthmodel.w]');
            obj.paras.PPP.states = rmfield(birthmodel,'w')';
            obj.paras.MBM.w = zeros(0,1);
            obj.paras.MBM.hypothesis_table = cell(0,1);
            obj.paras.MBM.tt = cell(0,1);
        end
        
        function obj = PMBM_predict(obj,motionmodel,birthmodel,P_S)
            obj = PPP_predict(obj,motionmodel,birthmodel,P_S);
            obj.paras.MBM.tt = cellfun(@(y) arrayfun(@(x) Bern_predict(obj,x,motionmodel,P_S), y), obj.paras.MBM.tt);
        end
        
        function obj = PMBM_update(obj)
            
        end
        
        function obj = recycle(obj)

        end
        
        function Bern = Bern_predict(obj,tt_entry,motionmodel,P_S)
            Bern = obj.paras.MBM.tt{tt_entry(1)}(tt_entry(2));
            Bern.r = Bern.r*P_S;
            Bern.state = obj.density.predict(Bern.state,motionmodel);
        end
        
        function [Bern, lik_undetected] = Bern_undetected_update(Bern,P_D,P_G)
            l_nodetect = Bern.r*(1 - P_D*P_G);
            lik_undetected = 1 - Bern.r + l_nodetect;
            Bern.r = l_nodetect/lik_undetected;
        end
        
        function [Bern, lik_detected] = Bern_detected_update(obj,tt_entry,z,measmodel,P_D)
            Bern = obj.paras.MBM.tt{tt_entry(1)}(tt_entry(2));
            lik_detected = obj.density.predictedLikelihood(Bern.state,z,measmodel) + log(P_D*Bern.r);
            Bern.state = obj.density.update(Bern.state,z,measmodel);
            Bern.r = 1;
        end
        
        function obj = PPP_predict(obj,motionmodel,birthmodel,P_S)
            obj.paras.PPP.w = obj.paras.PPP.w + log(P_S);
            obj.paras.PPP.states = arrayfun(@(x) obj.density.predict(x,motionmodel), obj.paras.PPP.states);
            obj.paras.PPP.w = [obj.paras.PPP.w;[birthmodel.w]'];
            obj.paras.PPP.states = [obj.paras.PPP.states;rmfield(birthmodel,'w')'];
        end
        
        function [Bern, lik_new] = PPP_detected_update(obj,z,measmodel,P_D,lambda_c)
            state_upd = arrayfun(@(x) obj.density.update(x,z,measmodel), obj.paras.PPP.states);
            w_upd = arrayfun(@(x) obj.density.predictedLikelihood(x,z,measmodel), obj.paras.PPP.states) + obj.paras.PPP.w + log(P_D);
            C = sum(exp(w_upd));
            lik_new = log(C + lambda_c);
            Bern.r = C/(C + lambda_c);
            Bern.state = obj.density.momentMatching(w_upd,state_upd);
        end
        
        function obj = PPP_undetected_update(obj,P_D,P_G)
            obj.paras.PPP.w = obj.paras.PPP.w + log(1 - P_D*P_G);
        end
        
        function obj = PPP_prune(obj,threshold)
            idx = obj.paras.PPP.w > threshold;
            obj.paras.PPP.w = obj.paras.PPP.w(idx);
            obj.paras.PPP.state = obj.paras.PPP.state(idx);
        end
        
        
    end
end

