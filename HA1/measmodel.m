classdef measmodel
    %MEASMODEL is a class containing different measurement models
    
    methods (Static)
        function obj = cv2Dmeasmodel(sigma)
            %CV2DMEASMODEL creates the measurement model for a 2D nearly
            %constant velocity motion model
            %INPUT:     sigma: standard deviation of measurement noise ---
            %scalar
            %OUTPUT:    obj.d: measurement dimension --- scalar
            %           obj.H: observation matrix --- 2 x 4
            %           matrix
            %           obj.D: measurement noise standard deviation --- 2 x
            %           2 matrix
            %           obj.R: measurement noise covariance --- 2 x 2
            %           matrix
            obj.d = 2;
            obj.H = [1 0 0 0;0 0 1 0];
            obj.D = sigma*eye(2);
            obj.R = obj.D*obj.D';
        end

    end
end

