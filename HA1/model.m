classdef model < handle
    %UNTITLED5 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties (Constant)
        T = 1
    end
    
    methods (Static)
        function obj = sensormodel(P_D,lambda_c,range_c)
            %UNTITLED Construct an instance of this class
            %   Detailed explanation goes here
            obj.P_D = P_D;
            obj.Q_D = 1-P_D;
            obj.lambda_c = lambda_c;
            obj.range_c = range_c;
            obj.pdf_c = 1/prod(range_c(:,2)-range_c(:,1));
        end
        
        function obj = groundtruth(nbirths,xstart,Pstart,tbirth,tdeath,K)
            %UNTITLED Construct an instance of this class
            %   Detailed explanation goes here
            obj.nbirths = nbirths;
            obj.xstart = xstart;
            obj.Pstart = Pstart;
            obj.tbirth = tbirth;
            obj.tdeath = tdeath;
            obj.K = K;
        end
         
    end

end

