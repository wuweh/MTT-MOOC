classdef measmodel
    %MEASMODEL is a class containing different measurement models
    
    methods (Static)
        function obj = cv2Dmeasmodel(sigma)
            %CV2DMEASMODEL creates the measurement model for a 2D nearly
            %constant velocity motion model
            %INPUT:     sigma: standard deviation of measurement noise ---
            %           scalar
            %OUTPUT:    obj.d: measurement dimension --- scalar
            %           obj.H: observation matrix --- 2 x 4
            %           matrix
            %           obj.R: measurement noise covariance --- 2 x 2
            %           matrix
            obj.d = 2;
            obj.H = @(x) [1 0 0 0;0 1 0 0];
            obj.R = sigma^2*eye(2);
            obj.h = @(x) obj.H(x)*x;
        end
        
        function obj = ct2Dmeasmodel(sigma)
            %CT2DMEASMODEL creates the measurement model for a 2D
            %coordinate turn motion model
            %INPUT:     sigma: standard deviation of measurement noise ---
            %           scalar
            %OUTPUT:    obj.d: measurement dimension --- scalar
            %           obj.H: observation matrix --- 2 x 5
            %           matrix
            %           obj.R: measurement noise covariance --- 2 x 2
            %           matrix
            % NOTE: the first two entries of the state vector represents
            % the X-position and Y-position, respectively.
            obj.d = 2;
            obj.H = @(x) [1 1 0 0 0;0 1 0 0 0];
            obj.R = sigma^2*eye(2);
            obj.h = @(x) obj.H(x)*x;
        end
        
        function obj = bearingmeasmodel(sigma, s)
            %BEARINGMEASUREMENT creats the bearing measurement model
            %INPUT: sigma: standard deviation of measurement noise ---
            %       scalar
            %       s: sensor position --- 2 x 1 vector
            %OUTPUT:obj.d: measurement dimension --- scalar
            %       obj.h: function handle to generate measurement ---
            %       scalar
            %       obj.H: function handle to call measurement model Jacobian
            %       --- 1 x (state dimension) vector
            %       obj.R: measurement noise covariance --- scalar
            %NOTE: the measurement model assumes that in the state vector
            %the first two entries are the X-position and Y-position,
            %respectively.
            
            obj.d = 1;
            %Range
            rng = @(x) norm(x(1:2)-s);
            %Bearing
            obj.h = @(x) atan2(x(2)-s(2),x(1)-s(1));
            %Measurement model Jacobian
            obj.H = @(x) [-(x(2)-s(2))/(rng(x)^2) (x(1)-s(1))/(rng(x)^2) zeros(1, length(x)-2)];
            %Measurement noise covariance
            obj.R = sigma^2;
            
        end
    end
end

