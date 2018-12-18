classdef multiobjectProcess
    %MULTIOBJECTPROCESS
    
    methods (Static)
        function instantiation = PoissonRFSs(lambda,spatial_distribution)
            %draw an integer v from Poisson distirbution with parameter
            %lambda
            v = poissrnd(lambda);
            %draw v elements from the given spatial distribution
            
        end

    end
end

