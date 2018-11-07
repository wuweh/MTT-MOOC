classdef singletargetracker
    %UNTITLED11 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        P_G
        gate_gamma
        x
        P
    end
    
    methods
        
        function obj = initiator(obj,P_G,m_d,x_0,P_0)
            obj.P_G = P_G;
            obj.gate_gamma = chi2inv(P_G,m_d);
            obj.x = x_0;
            obj.P = P_0;
        end
        
        function obj = linearKalmanPredict(obj, motionmodel)
            obj.x = motionmodel.A*obj.x;
            obj.P = motionmodel.Q+motionmodel.A*obj.P*motionmodel.A';
        end
        
        function obj = nearestNeighborLinearKalmanUpdate(obj, z, measmodel)
            % measurement uncertainty
            S = measmodel.H*obj.P*measmodel.H' + measmodel.R;
            zlength = size(z,2);
            nu = z - measmodel.H*repmat(obj.x,[1 zlength]);
            dist= sum((inv(chol(S))'*nu).^2);
            
            if min(dist) < obj.gate_gamma
                K = obj.P*measmodel.H'/S;
                temp = eye(size(obj.x,1)) - K*measmodel.H;
                obj.P = temp*obj.P*temp' + K*measmodel.R*K';
                [~,nn_idx] = min(dist);
                obj.x = obj.x + K*nu(:,nn_idx);
            end
        end
        
        function obj = probDataAssocLinearKalmanUpdate(obj, z, measmodel, sensormodel)
            % measurement uncertainty
            S = measmodel.H*obj.P*measmodel.H' + measmodel.R;
            zlength = size(z,2);
            nu = z - measmodel.H*repmat(obj.x,[1 zlength]);
            dist= sum((inv(chol(S))'*nu).^2);
            
            if min(dist) < obj.gate_gamma
                K = obj.P*measmodel.H'/S;
                s_d = size(obj.x,1);
                temp = eye(s_d) - K*measmodel.H;
                
                meas_update_idx = find(dist < obj.gate_gamma);
                num_meas_ingate = length(meas_update_idx);
                
                mu = zeros(num_meas_ingate+1,1);
                x_temp = zeros(s_d,num_meas_ingate+1);
                P_temp = zeros(s_d,s_d,num_meas_ingate+1);
                
                % missed detection
                mu(1) = (1-sensormodel.P_D*obj.P_G)*(sensormodel.lambda_c)*(sensormodel.pdf_c);
                x_temp(:,1) = obj.x;
                P_temp(:,:,1) = obj.P;
                % measurements in the gate
                for i = 1:num_meas_ingate
                    mu(i+1) = mvnpdf(z(:,meas_update_idx(i)),measmodel.H*obj.x,S);
                    x_temp(:,i+1) = obj.x + K*(z(:,meas_update_idx(i))-measmodel.H*obj.x);
                    P_temp(:,:,i+1) = temp*obj.P*temp' + K*measmodel.R*K';
                end
                % normalize
                mu = mu/sum(mu);
                z_eq = mu(1)*measmodel.H*obj.x + z(:,meas_update_idx)*mu(2:end);
                obj.x = obj.x + K*(z_eq-measmodel.H*obj.x);
                for i = 1:num_meas_ingate+1
                    temp = x_temp(:,i)-obj.x;
                    obj.P = mu(i)*P_temp(:,:,i) + mu(i)*(temp*temp');
                end
            end
            
        end
        
    end
end

