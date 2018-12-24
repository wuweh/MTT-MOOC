classdef multiobjectProcess
    %MULTIOBJECTPROCESS
    
    properties
        %only 1D and 2D distribution are considered
        spatial_distribution
    end
    
    methods
        
        function obj = spatialDistribution(obj,inputArg1,inputArg2,inputArg3)
            %SPATIALDISTRIBUTION constructs an instance of this class
            %depending on inputs, create the corresponding distribution
            switch nargin
                case 2
                    obj.spatial_distribution.name = 'u';
                    obj.spatial_distribution.paras{1} = inputArg1;
                case 3
                    obj.spatial_distribution.name = 'g';
                    obj.spatial_distribution.paras{1} = inputArg1;
                    obj.spatial_distribution.paras{2} = inputArg2;
                case 4
                    obj.spatial_distribution.name = 'gm';
                    obj.spatial_distribution.paras{1} = inputArg1;
                    obj.spatial_distribution.paras{2} = inputArg2;
                    obj.spatial_distribution.paras{3} = inputArg3;
            end
        end
        
        function instantiation = PoissonRFSs(obj,lambda)
            %draw an integer v from Poisson distirbution with parameter
            %lambda
            v = poissrnd(lambda);
            %draw v elements from the given spatial distribution
            switch obj.spatial_distribution.name
                case 'u'
                    instantiation = unifrnd(obj.spatial_distribution.paras{1}(:,1),obj.spatial_distribution.paras{1}(:,2),v,1);
                case 'g'
                    instantiation = mvnrnd(obj.spatial_distribution.paras{1},obj.spatial_distribution.paras{2},v);
                case 'gm'
                    gm = gmdistribution(obj.spatial_distribution.paras{1},obj.spatial_distribution.paras{2},obj.spatial_distribution.paras{3});
                    instantiation = random(gm,v,1);
            end
        end
        
        function instantiation = BernoulliRFSs(obj,r)
            %draw an integer v from Bernoulli distirbution with parameter r
            v = binornd(1,r);
            %draw v elements from the given spatial distribution
            switch obj.spatial_distribution.name
                case 'u'
                    instantiation = unifrnd(obj.spatial_distribution.paras{1}(:,1),obj.spatial_distribution.paras{1}(:,2),v,1);
                case 'g'
                    instantiation = mvnrnd(obj.spatial_distribution.paras{1},obj.spatial_distribution.paras{2},v);
                case 'gm'
                    gm = gmdistribution(obj.spatial_distribution.paras{1},obj.spatial_distribution.paras{2},obj.spatial_distribution.paras{3});
                    instantiation = random(gm,v,1);
            end
        end

    end
end

