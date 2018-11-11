classdef motionmodel
    %MOTIONMODEL is a class containing different motion models
    
    methods (Static)
        function obj = cv2Dmodel(T,sigma)
            %CV2DMODEL creates a 2D nearly constant velocity model
            %INPUT:     T: sampling time --- scalar
            %           sigma: standard deviation of motion noise --- scalar
            %OUTPUT:    obj.d: target state dimension --- scalar
            %           obj.A: motion transition matrix --- 2 x 2 matrix
            %           obj.Q: motion noise covariance --- 4 x 4 matrix
            %           obj.f: state prediction function handle
            obj.d = 4;
            A1dim = [1 T; 0 1];
            Q1dim = sigma^2*[T^4/4 T^3/2; T^3/2 T^2];
            
            obj.A = blkdiag(A1dim,A1dim);
            obj.Q = blkdiag(Q1dim,Q1dim);
            
            obj.f = @(x) obj.A*x;
        end
        
        function obj = ct2Dmodel(T,sigmaV,sigmaOmega)
            %CT2DMODEL creates a 2D coordinate turn model with nearly constant
            %polar velocity and turn rate
            %INPUT:     T: sampling time --- scalar
            %           sigmaV: standard deviation of motion noise added to
            %           polar velocity --- scalar
            %           sigmaOmega: standard deviation of motion noise added to
            %           turn rate --- scalar
            %OUTPUT:    obj.d: target state dimension --- scalar
            %           obj.F: motion Jacobian matrix --- 5 x 5 matrix
            %           obj.f: state prediction function handle
            %           obj.Q: motion noise covariance --- 5 x 5 matrix
            % NOTE: the motion model assumes that the state vector x consist of the
            % following states:
            %           px          X-position
            %           py          Y-position
            %           v           velocity
            %           phi         heading
            %           omega       turn-rate
            obj.d = 5;
            obj.f = @(x) x + [
                T*x(3)*cos(x(4));
                T*x(3)*sin(x(4));
                0;
                T*x(5);
                0];
            obj.F = @(x) [
                1 0 T*cos(x(4)) -T*x(3)*sin(x(4)) 0;
                0 1 T*sin(x(4)) T*x(3)*cos(x(4))  0;
                0 0 1           0                 0;
                0 0 0           1                 T;
                0 0 0           0                 1
                ];
            G = [zeros(2,2); 1 0; 0 0; 0 1];
            obj.Q = G*diag([sigmaV^2 sigmaOmega^2])*G';
        end

    end
end


