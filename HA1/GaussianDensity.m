classdef GaussianDensity < matlab.mixin.Copyable
    
    properties
        
        %Mean vector
        x
        %Covariance matrix
        P
        
    end
    
    methods
        
        function initialize(obj,x_0,P_0)
            
            %Mean vector and covariance matrix
            obj.x = x_0;
            obj.P = P_0;
            
        end
        
        function parameters = returnParameters(obj)
            
            %Mean vector
            parameters.x = obj.x;
            %Covariance matrix
            parameters.P = obj.P;
            
        end
        
        function setParameters(obj,parameters)
            
            %Mean vector
            obj.x = parameters.x;
            %Covariance matrix
            obj.P = parameters.P;
            
        end
        
        function KalmanPredict(obj,motionmodel)
            obj.x = motionmodel.f(obj.x);
            obj.P = motionmodel.F(obj.x)*obj.P*motionmodel.F(obj.x)'+motionmodel.Q;
            %Make sure P is symmetric
            %obj.P = (obj.P+obj.P')/2;
        end
        
        function [meas_likelihood] = KalmanUpdate(obj,z,measmodel)
            %Measurement model Jacobian
            Hx = measmodel.H(obj.x);
            %Innovation covariance
            S = Hx*obj.P*Hx' + measmodel.R;
            %Make sure matrix S is positive definite
            S = (S+S')/2;
            
            %Use choleskey decomposition to speed up matrix inversion
            Vs = chol(S);
            inv_sqrt_S = inv(Vs);
            iS= inv_sqrt_S*inv_sqrt_S';
            K  = obj.P*Hx'*iS;
            %K = (P*Hx')/S;
            
            %Predicted observation
            z_hat = measmodel.h(obj.x);
            %State update
            obj.x = obj.x + K*(z - z_hat);
            %Covariance update
            obj.P = (eye(size(obj.x,1)) - K*Hx)*obj.P;
            %Make sure P_upd is symmetric
            %obj.P = 0.5*(obj.P + obj.P');
            
            det_S = prod(diag(Vs))^2;
            %Calculate predicted likelihood
            meas_likelihood = -0.5*(log(2*pi*det_S)+dot(z-z_hat,iS*(z-z_hat)));
        end
        
        function [meas_likelihood] = measLikelihood(obj,z,measmodel)
            %Measurement model Jacobian
            Hx = measmodel.H(obj.x);
            %Innovation covariance
            S = Hx*obj.P*Hx' + measmodel.R;
            %Make sure matrix S is positive definite
            S = (S+S')/2;
            %Calculate predicted likelihood
            meas_likelihood = log_mvnpdf(z',(measmodel.h(obj.x))',S);
        end
        
        function z_ingate = ellipsoidalGating(obj,z,measmodel,gating_size)
            zlength = size(z,2);
            in_gate = false(zlength,1);
            
            %Measurement model Jacobian
            Hx = measmodel.H(obj.x);
            
            S = Hx*obj.P*Hx' + measmodel.R;
            %Make sure matrix S is positive definite
            S = (S+S')/2;
            %Use choleskey decomposition to speed up matrix decomposition
            [Vs,~] = chol(S);
            
            nu = z - repmat(measmodel.h(obj.x),[1 zlength]);
            dist= sum((inv(Vs)'*nu).^2);
            
            in_gate(dist<gating_size) = true;
            z_ingate = z(:,in_gate);
        end
        
        function x = expectedValue(obj)
            
            %Expected value
            x = obj.x;
        end
        
        function P = covariance(obj)
            
            %Covariance
            P = obj.P;
        end
        
    end
end