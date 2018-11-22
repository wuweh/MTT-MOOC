classdef hypothesisUpdate

    methods (Static)
        
        function w_miss = undetected(P_D,P_G)
            w_miss = log(1-P_D*P_G);
        end
        
        function [state,w_upd] = detected(density,state,z,measmodel,P_D)
            % Updated state density, compute log likelihood of measurement
            state = density.update(state,z,measmodel);
            % Updated log likelihood
            meas_likelihood = density.measLikelihood(state, z, measmodel);
            w_upd = meas_likelihood + log(P_D);
        end
        
    end
    
end