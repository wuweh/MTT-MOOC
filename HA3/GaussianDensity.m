classdef GaussianDensity
    
    methods (Static)
        
        function expected_value = expectedValue(density)
            expected_value = density.x;
        end
        
        function covariance = covariance(density)
            covariance = density.P;
        end
        
        function density_pred = predict(density, motionmodel)
            %PREDICT performs linear/nonlinear (Extended) Kalman prediction step
            %INPUT: state: a structure with two fields:
            %                   x: object state mean --- (state dimension) x 1 vector
            %                   P: object state covariance --- (state dimension) x (state dimension) matrix
            %       motionmodel: a structure specifies the motion model parameters
            %OUTPUT:state_pred: a structure with two fields:
            %                   x: predicted object state mean --- (state dimension) x 1 vector
            %                   P: predicted object state covariance --- (state dimension) x (state dimension) matrix
            
            density_pred.x = motionmodel.f(density.x);
            density_pred.P = motionmodel.F(density.x)*density.P*motionmodel.F(density.x)'+motionmodel.Q;
            
        end
        
        function density_upd = update(density_pred, z, measmodel)
            %UPDATE performs linear/nonlinear (Extended) Kalman update step
            %INPUT: z: measurements --- (measurement dimension) x 1 vector
            %       state_pred: a structure with two fields:
            %                   x: predicted object state mean --- (state dimension) x 1 vector
            %                   P: predicted object state covariance --- (state dimension) x (state dimension) matrix
            %       measmodel: a structure specifies the measurement model parameters
            %OUTPUT:state_upd: a structure with two fields:
            %                   x: updated object state mean --- (state dimension) x 1 vector
            %                   P: updated object state covariance --- (state dimension) x (state dimension) matrix
            
            %Measurement model Jacobian
            Hx = measmodel.H(density_pred.x);
            %Innovation covariance
            S = Hx*density_pred.P*Hx' + measmodel.R;
            %Make sure matrix S is positive definite
            S = (S+S')/2;
            
            %Use choleskey decomposition to speed up matrix inversion
            %             Vs = chol(S);
            %             inv_sqrt_S = inv(Vs);
            %             iS= inv_sqrt_S*inv_sqrt_S';
            %             K  = state_pred.P*Hx'*iS;
            K = (density_pred.P*Hx')/S;
            
            %State update
            density_upd.x = density_pred.x + K*(z - measmodel.h(density_pred.x));
            %Covariance update
            density_upd.P = (eye(size(density_pred.x,1)) - K*Hx)*density_pred.P;
            
        end
        
        function predict_likelihood = predictedLikelihood(density_pred,z,measmodel)
            %PREDICTLIKELIHOOD calculates the predicted likelihood in logarithm domain
            %INPUT:  z: measurements --- (measurement dimension) x (number of measurements) matrix
            %        state_pred: a structure with two fields:
            %                   x: predicted object state mean --- (state dimension) x 1 vector
            %                   P: predicted object state covariance --- (state dimension) x (state dimension) matrix
            %        measmodel: a structure specifies the measurement model parameters
            %OUTPUT: meas_likelihood: measurement update likelihood for each measurement in logarithm domain --- (number of measurements) x 1 vector
            %Measurement model Jacobian
            Hx = measmodel.H(density_pred.x);
            %Innovation covariance
            S = Hx*density_pred.P*Hx' + measmodel.R;
            %Make sure matrix S is positive definite
            S = (S+S')/2;
            %Calculate predicted likelihood
            predict_likelihood = log_mvnpdf(z',(measmodel.h(density_pred.x))',S);
        end
        
        function [z_ingate, meas_in_gate] = ellipsoidalGating(density_pred, z, measmodel, gating_size)
            %ELLIPSOIDALGATING performs ellipsoidal gating for a single target
            %INPUT:  z: measurements --- (measurement dimension) x (number of measurements) matrix
            %        state_pred: a structure with two fields:
            %                   x: predicted object state mean --- (state dimension) x 1 vector
            %                   P: predicted object state covariance --- (state dimension) x (state dimension) matrix
            %        measmodel: a structure specifies the measurement model parameters
            %        gating_size: gating size --- scalar
            %OUTPUT: z_ingate: measurements in the gate --- (measurement dimension) x (number of measurements in the gate) matrix
            %        meas_in_gate: boolean vector indicating whether the
            %        corresponding measurement is in the gate or not ---
            %        (number of measurements) x 1
            if isempty(z)
                z_ingate = z;
                meas_in_gate = false(0,1);
                return
            end
            zlength = size(z,2);
            meas_in_gate = false(zlength,1);
            
            S = measmodel.H(density_pred.x)*density_pred.P*measmodel.H(density_pred.x)' + measmodel.R;
            %Make sure matrix S is positive definite
            S = (S+S')/2;
            
            nu = z - repmat(measmodel.h(density_pred.x),[1 zlength]);
            dist = diag(nu.'*(S\nu));
            
            meas_in_gate(dist<gating_size) = true;
            z_ingate = z(:,meas_in_gate);
        end
        
        function density = momentMatching(w, mixture_density)
            %MOMENTMATCHING: approximate a Gaussian mixture density as a single Gaussian using moment matching
            %INPUT: w: normalised weight of Gaussian components in logarithm domain --- (number of Gaussians) x 1 vector
            %       states: structure array of size (number of Gaussian components x 1), each structure has two fields
            %               x: means of Gaussian components --- (variable dimension) x (number of Gaussians) matrix
            %               P: variances of Gaussian components --- (variable dimension) x (variable dimension) x (number of Gaussians) matrix
            %OUTPUT:state: a structure with two fields:
            %               x_hat: approximated mean --- (variable dimension) x 1 vector
            %               P_hat: approximated covariance --- (variable dimension) x (variable dimension) matrix
            
            if length(w) == 1
                density = mixture_density;
                return;
            end
            
            w = exp(w);
            %Moment matching
            density.x = [mixture_density(:).x]*w;
            numGaussian = length(w);
            density.P = zeros(size(mixture_density(1).P));
            for i = 1:numGaussian
                %Add spread of means
                x_diff = mixture_density(i).x - density.x;
                density.P = density.P + w(i).*(mixture_density(i).P + x_diff*x_diff');
            end
        end
        
        function [w_hat,mixture_density_hat] = mixtureReduction(w,mixture_density,threshold)
            %MIXTUREREDUCTION: uses a greedy merging method to reduce the number of Gaussian components for a Gaussian mixture density
            %INPUT: w: normalised weight of Gaussian components in logarithm domain --- (number of Gaussians) x 1 vector
            %       states: structure array of size (number of Gaussian components x 1), each structure has two fields
            %               x: means of Gaussian components --- (variable dimension) x (number of Gaussians) matrix
            %               P: variances of Gaussian components --- (variable dimension) x (variable dimension) x (number of Gaussians) matrix
            %INPUT: w_hat: normalised weight of Gaussian components in logarithm domain after merging--- (number of Gaussians) x 1 vector
            %       states_hat: structure array of size (number of Gaussian components after merging x 1), each structure has two fields
            %               x: means of Gaussian components --- (variable dimension) x (number of Gaussians after merging) matrix
            %               P: variances of Gaussian components --- (variable dimension) x (variable dimension) x (number of Gaussians after merging) matrix
            %       threshold: merging threshold --- scalar
            
            if length(w) == 1
                w_hat = w;
                mixture_density_hat = mixture_density;
                return;
            end
            
            %Index set of components
            I = 1:length(mixture_density);
            el = 1;
            
            while ~isempty(I)
                Ij = [];
                %Find the component with the highest weight
                [~,j] = max(w);
                
                for i = I
                    temp = mixture_density(i).x-mixture_density(j).x;
                    val = diag(temp.'*(mixture_density(j).P\temp));
                    %Find other similar components in the sense of small Mahalanobis distance
                    if val <= threshold
                        Ij= [ Ij i ];
                    end
                end
                
                %Merge components by moment matching
                [temp,w_hat(el,1)] = normalizeLogWeights(w(Ij));
                mixture_density_hat(el,1) = GaussianDensity.momentMatching(temp, mixture_density(Ij));
                
                %Remove indices of merged components from index set
                I = setdiff(I,Ij);
                %Set a negative to make sure this component won't be selected again
                w(Ij,1) = log(eps);
                el = el+1;
            end
            
            %Normalize the weights
%             [w_hat,~] = normalizeLogWeights(w_hat);
            
        end
        
    end
end