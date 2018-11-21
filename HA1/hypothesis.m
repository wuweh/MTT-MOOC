classdef hypothesis < matlab.mixin.Copyable
    
    properties
        
        % State density
        density = [];
        % Log likelihood
        log_lik = [];
        
    end
    
    methods
        
        function initialize(obj,density_class_handle,x_0,P_0)
            
            % Initialize class instance for density
            obj.density = feval(density_class_handle);
            % Initialize density
            obj.density.initialize(x_0,P_0);
            % Initialize log likelihood
            obj.log_lik = 0;
            
        end
        
        function parameters = returnParameters(obj)
            
            % Returns the density parameters
            parameters = obj.density.returnParameters;
            
        end
        
        function expected_value = expectedValue(obj)
            
            % Expected value of the state
            expected_value = obj.density.expectedValue;
            
        end
        
        function predict(obj,motionmodel)
            
            obj.density.KalmanPredict(motionmodel);
            
        end
        
        function missDetectHypothesis(obj,P_D,P_G)
            
            w_miss = log(1-P_D*P_G);
            % Updated log likelihood
            obj.log_lik = obj.log_lik + w_miss;
        end
        
        function measUpdateHypothesis(obj,z,measmodel,P_D)
            
            % Updated state density, compute log likelihood of measurement
            [meas_likelihood] = obj.density.KalmanUpdate(measmodel,z);
            % Updated log likelihood
            obj.log_lik = obj.log_lik + meas_likelihood + log(P_D);
            
        end
        
        
    end
    
end