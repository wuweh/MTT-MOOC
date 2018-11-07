classdef motionmodel < model
    %UNTITLED4 Summary of this class goes here
    %   Detailed explanation goes here
    
    methods (Static)
        function obj = cv2Dmodel(sigma)
            %UNTITLED4 Construct an instance of this class
            %   Detailed explanation goes here
            obj.d = 4;
            T = model.T;
            A1dim = [1 T; 0 1];
            Q1dim = sigma^2*[T^4/4 T^3/2; T^3/2 T^2];
            obj.A = blkdiag(A1dim,A1dim);
            obj.Q = blkdiag(Q1dim,Q1dim);
        end

    end
end


