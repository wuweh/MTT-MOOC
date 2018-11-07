classdef measmodel < model
    %UNTITLED6 Summary of this class goes here
    %   Detailed explanation goes here
    
    methods (Static)
        function obj = cv2Dmeasmodel(sigma)
            %UNTITLED6 Construct an instance of this class
            %   Detailed explanation goes here
            obj.d = 2;
            obj.H = [1 0 0 0;0 0 1 0];
            obj.R = sigma^2*eye(2);
        end

    end
end

