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
            obj.d = 4;
            A1dim = [1 T; 0 1];
            Q1dim = sigma^2*[T^4/4 T^3/2; T^3/2 T^2];
            obj.A = blkdiag(A1dim,A1dim);
            obj.Q = blkdiag(Q1dim,Q1dim);
        end

    end
end


