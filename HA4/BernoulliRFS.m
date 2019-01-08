classdef BernoulliRFS
    
    properties
        density %density class handle
    end
    
    methods
        function [obj,Bern] = initialize(obj,density_class_handle,r,state)
            obj.density = density_class_handle;
            Bern.r = r;
            Bern.state = state;
        end
        
        function Bern = predict(obj,Bern,motionmodel,P_S)
            Bern.r = Bern.r*P_S;
            Bern.state = obj.density.predict(Bern.state,motionmodel);
        end
        
        function [Bern, lik] = update_undetected(Bern,P_D,P_G)
            l_nodetect = Bern.r*(1 - P_D*P_G);
            lik = 1 - Bern.r + l_nodetect;
            Bern.r = l_nodetect/lik;
        end
        
        function [Bern, lik] = update_detected(obj,Bern,z,measmodel,P_D)
            lik = obj.density.predictedLikelihood(Bern.state,z,measmodel) + log(P_D*Bern.r);
            Bern.state = obj.density.update(Bern.state,z,measmodel);
            Bern.r = 1;
        end
        
    end
end

