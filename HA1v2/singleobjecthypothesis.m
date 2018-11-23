classdef singleobjecthypothesis

    methods (Static)
        
        function w_miss = undetected(P_D,P_G)
            %UNDETECTED calculates the missed detection likelihood
            %INPUT: P_D: object detection probability --- scalar
            %       P_G: gating size in percentage --- scalar
            %OUTPUT:w_miss: missed detection likelihood in logarithm --- scalar
            w_miss = log(1-P_D*P_G);
        end
        
        function [state_upd,w_upd] = detected(density,state,z,measmodel,P_D)
            %DETECTED creates a measurement update hypothesis
            %INPUT: density: a class handle
            %       state: a structure with two fields:
            %                x: object state mean --- (state dimension) x 1 vector
            %                P: object state covariance --- (state dimension) x (state dimension) matrix
            %       z: measurement --- (measurement dimension x 1) vector
            %       measmodel: a structure specifies measurement model
            %       P_D: object detection probability --- scalar
            %OUTPUT:state_upd: a structure array of size (number of measurements x 1) with two fields:
            %                x: updated state mean --- (state dimension) x 1 vector
            %                P: updated state covariance --- (state dimension) x (state dimension) matrix
            %       w_upd: measurement update likelihood in logarithm --- (number of measurements x 1) vector
            
            % Updated log likelihood
            predict_likelihood = density.predictedLikelihood(state, z, measmodel);
            
            % Updated state density, compute predicted likelihood
            for i = 1:size(z,2)
                state_upd(i,1) = density.update(state,z(:,i),measmodel);
            end
            
            w_upd = predict_likelihood + log(P_D);
        end
        
    end
    
end