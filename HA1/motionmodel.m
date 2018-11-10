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
            %           obj.B: noise transition matrix --- 4 x 2 matrix
            obj.d = 4;
            A0 = [1 T; 0 1];                   
            obj.A = blkdiag(A0,A0);
            B0 = [(T^2)/2; T];
            obj.B = sigma*blkdiag(B0,B0);
            obj.Q= obj.B*obj.B';
        end

    end
end


