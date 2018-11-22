classdef GaussianDensity
    
    methods (Static)
        
        function state_pred = predict(state, motionmodel)
            %KALMANPREDICT performs linear/nonlinear (Extended) Kalman prediction step
            %INPUT: x: target state --- (state dimension) x 1 vector
            %       P: target state covariance --- (state dimension) x (state
            %           dimension) matrix
            %       motionmodel: a structure specifies the motion model parameters:
            %                   F: function handle return transition/Jacobian matrix
            %                   f: function handle return predicted targe state
            %                   Q: motion noise covariance matrix
            %OUTPUT:x_pred: predicted target state --- (state dimension) x 1 vector
            %       P_pred: predicted target state covariance --- (state dimension) x (state
            %           dimension) matrix
            
            state_pred.x = motionmodel.f(state.x);
            state_pred.P = motionmodel.F(state.x)*state.P*motionmodel.F(state.x)'+motionmodel.Q;
            
        end
        
        function state_upd = update(state_pred, z, measmodel)
            %KALMANUPDATE performs linear/nonlinear (Extended) Kalman update step
            %INPUT: z: measurements --- (measurement dimension) x 1 vector
            %       x: target state --- (state dimension) x 1 vector
            %       P: target state covariance --- (state dimension) x (state
            %           dimension) matrix
            %       measmodel: a structure specifies the measurement model parameters:
            %                   H: function handle return transition/Jacobian matrix
            %                   h: function handle return the observation of the target state
            %                   R: measurement noise covariance matrix
            %OUTPUT:x_upd: updated target state --- (state dimension) x 1 vector
            %       P_upd: updated target state covariance --- (state dimension) x (state
            %           dimension) matrix
            
            %Measurement model Jacobian
            Hx = measmodel.H(state_pred.x);
            %Innovation covariance
            S = Hx*state_pred.P*Hx' + measmodel.R;
            %Make sure matrix S is positive definite
            S = (S+S')/2;
            
            %Use choleskey decomposition to speed up matrix inversion
%             Vs = chol(S);
%             inv_sqrt_S = inv(Vs);
%             iS= inv_sqrt_S*inv_sqrt_S';
%             K  = state_pred.P*Hx'*iS;
            K = (state_pred.P*Hx')/S;
            
            %State update
            state_upd.x = state_pred.x + K*(z - measmodel.h(state_pred.x));
            %Covariance update
            state_upd.P = (eye(size(state_pred.x,1)) - K*Hx)*state_pred.P;
            
        end
        
        function meas_likelihood = measLikelihood(state_pred,z,measmodel)
            %MEASLIKELIHOOD calculates the predicted likelihood in logarithm domain, i.e., N(z;\bar{z},S).
            %INPUT:  z: measurements --- (measurement dimension) x (number
            %           of measurements) matrix
            %        x: target state mean --- (target state dimension) x 1 vector
            %        P: target state covariance --- (target state dimension) x (target state dimension) matrix
            %       measmodel: a structure specifies the measurement model parameters
            %           d: measurement dimension --- scalar
            %           H: function handle return transition/Jacobian matrix
            %           h: function handle return the observation of the target
            %                   state
            %           R: measurement noise covariance matrix
            %OUTPUT: meas_likelihood: measurement update likelihood for
            %       each measurement in logarithm domain --- (number of measurements) x 1 vector
            %Measurement model Jacobian
            Hx = measmodel.H(state_pred.x);
            %Innovation covariance
            S = Hx*state_pred.P*Hx' + measmodel.R;
            %Make sure matrix S is positive definite
            S = (S+S')/2;
            %Calculate predicted likelihood
            meas_likelihood = log_mvnpdf(z',(measmodel.h(state_pred.x))',S);
        end
        
        function z_ingate = ellipsoidalGating(state_pred, z, measmodel, gating_size)
            %GATING performs ellipsoidal gating for a single target
            %INPUT:  z: measurements --- (measurement dimension) x (number
            %           of measurements) matrix
            %        x: target state --- (state dimension) x 1 vector
            %        P: target state covariance --- (state dimension) x (state
            %           dimension) matrix
            %        measmodel: a structure specifies the measurement model parameters
            %        gating_size: gating size --- scalar
            %OUTPUT: z_ingate: measurements in the gate --- (measurement
            %                   dimension) x (number of measurements in the gate) matrix
            zlength = size(z,2);
            in_gate = false(zlength,1);
            
            S = measmodel.H(state_pred.x)*state_pred.P*measmodel.H(state_pred.x)' + measmodel.R;
            %Make sure matrix S is positive definite
            S = (S+S')/2;
            %Use choleskey decomposition to speed up matrix decomposition
%             [Vs,~] = chol(S);
            
            nu = z - repmat(measmodel.h(state_pred.x),[1 zlength]);
%             dist= sum((inv(Vs)'*nu).^2);
            
            dist = diag(nu.'*(S\nu));
            
            in_gate(dist<gating_size) = true;
            z_ingate = z(:,in_gate);
            
        end
        
        function state = momentMatching(w, states)
            %GAUSSIANMIXTUREREDUCTION: approximate a Gaussian mixture
            %density as a single Gaussian using a greedy approach
            %INPUT: w: normalised weight of Gaussian components in logarithm domain --- (number
            %           of Gaussians) x 1 vector
            %       x: means of Gaussian components --- (variable dimension)
            %           x (number of Gaussians) matrix
            %       P: variances of Gaussian components --- (variable
            %           dimension) x (variable dimension) x (number of Gaussians) matrix
            %OUTPUT:x_hat: approximated mean --- (variable dimension) x 1
            %vector
            %       P_hat: approximated covariance --- (variable dimension)
            %               x (variable dimension) matrix
            
            if length(w) == 1
                state = states;
                return;
            end
            
            w = exp(w);
            %Moment matching
            state.x = [states(:).x]*w;
            numGaussian = length(w);
            state.P = zeros(size(states(1).P));
            for i = 1:numGaussian
                %Add spread of means
                x_diff = states(i).x - state.x;
                state.P = state.P + w(i).*(states(i).P + x_diff*x_diff');
            end
        end
        
        function [w_hat,states_hat] = mixtureReduction(w,states,threshold)
            %MERGE merges hypotheses within small Mahalanobis distance
            %INPUT: hypothesesWeight: the weights of different hypotheses in logarithm domain ---
            %                       (number of hypotheses) x 1 vector
            %       multiHypotheses: (number of hypotheses) x 1 structure
            %                       with two fields: x: target state mean;
            %                       P: target state covariance
            %       threshold: hypotheses with Mahalanobis distance smaller this
            %                   value will be merged --- scalar
            %OUTPUT:hypothesesWeightMerged: hypotheses weights after merging in logarithm domain ---
            %                               (number of hypotheses after merging) x 1 vector
            %       multiHypothesesMerged: (number of hypotheses after merging) x 1 structure
            
            if length(w) == 1
                w_hat = w;
                states_hat = states;
                return;
            end

            %Index set of hypotheses
            I = 1:length(states);
            el = 1;
            
            while ~isempty(I)
                Ij = [];
                %Find the hypothesis with the highest weight
                [~,j] = max(w);
                [Vp,~] = chol(states(j).P);
                
                for i = I
                    temp = states(i).x-states(j).x;
                    val= sum((inv(Vp)'*temp).^2);
                    %Find other similar hypotheses in the sense of small Mahalanobis distance
                    if val <= threshold
                        Ij= [ Ij i ];
                    end
                end
                
                %Merge hypotheses (weighted average) within small Mahalanobis distance
                [temp,w_hat(el,1)] = normalizeLogWeights(w(Ij));
                states_hat(el,1) = GaussianDensity.momentMatching(temp, states(Ij));
                
                %Remove indices of merged hypotheses from hypotheses index set
                I = setdiff(I,Ij);
                %Set a negative to make sure this hypothesis won't be selected again
                w(Ij,1) = log(eps);
                el = el+1;
            end
            
            %Normalize the weights
            [w_hat,~] = normalizeLogWeights(w_hat);
            
        end
        
    end
end