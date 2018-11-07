classdef modelgen < handle
    %MODELGEN is a class used to generate the tracking model
    
    methods (Static)
        function obj = sensormodel(P_D,lambda_c,range_c)
            %SENSORMODEL generates the sensor model
            %INPUT:     P_D: target detection probability --- scalar
            %           lambda_c: average number of clutter measurements
            %           --- scalar
            %           per time scan, Poisson distributed --- scalar
            %           range_c: range of surveillance area --- 2 x 2
            %           matrix of the form [xmin xmax;ymin ymax]
            %OUTPUT:    obj.pdf_c: clutter (Poisson) intensity --- scalar
            obj.P_D = P_D;
            obj.lambda_c = lambda_c;
            obj.range_c = range_c;
            obj.pdf_c = 1/prod(range_c(:,2)-range_c(:,1));
        end
        
        function obj = groundtruth(nbirths,xstart,tbirth,tdeath,K)
            %GROUNDTRUTH specifies the parameters to generate groundtruth
            %INPUT:     nbirths: number of targets to be tracked --- scalar
            %           xstart: target initial states --- (target state
            %           dimension) x nbirths matrix
            %           tbirth: time step when targets are born --- (target state
            %           dimension) x 1 vector 
            %           tdeath: time step when targets die --- (target state
            %           dimension) x 1 vector
            %           K: total tracking time --- scalar
            obj.nbirths = nbirths;
            obj.xstart = xstart;
            obj.tbirth = tbirth;
            obj.tdeath = tdeath;
            obj.K = K;
        end
         
    end

end

