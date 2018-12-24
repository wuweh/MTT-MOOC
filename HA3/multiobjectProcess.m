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
                    for i = 1:length(inputArg1)
                        obj.spatial_distribution.paras{1}{i} = inputArg1{i};
                    end
                case 3
                    obj.spatial_distribution.name = 'g';
                    for i = 1:length(inputArg1)
                        obj.spatial_distribution.paras{1}{i} = inputArg1{i};
                        obj.spatial_distribution.paras{2}{i} = inputArg2{i};
                    end
                case 4
                    obj.spatial_distribution.name = 'gm';
                    for i = 1:length(inputArg1)
                        obj.spatial_distribution.paras{1}{i} = inputArg1{i};
                        obj.spatial_distribution.paras{2}{i} = inputArg2{i};
                        obj.spatial_distribution.paras{3}{i} = inputArg3{i};
                    end
            end
        end
        
        function instantiation = drawSamples(obj,idxParas,v)
            %draw v elements from the given spatial distribution
            switch obj.spatial_distribution.name
                case 'u'
                    instantiation = unifrnd(obj.spatial_distribution.paras{1}{idxParas}(:,1),obj.spatial_distribution.paras{1}{idxParas}(:,2),v,1);
                case 'g'
                    instantiation = mvnrnd(obj.spatial_distribution.paras{1}{idxParas},obj.spatial_distribution.paras{2}{idxParas},v);
                case 'gm'
                    gm = gmdistribution(obj.spatial_distribution.paras{1}{idxParas},obj.spatial_distribution.paras{2}{idxParas},obj.spatial_distribution.paras{3}{idxParas});
                    instantiation = random(gm,v,1);
            end
        end
        
        function instantiation = PoissonRFSs(obj,lambda)
            %draw an integer v from Poisson distirbution with parameter
            %lambda
            v = poissrnd(lambda);
            %draw v elements from the given spatial distribution
            instantiation = drawSamples(obj,1,v);
        end
        
        function instantiation = BernoulliRFSs(obj,r)
            %draw an integer v from Bernoulli distirbution with parameter r
            v = binornd(1,r);
            %draw v elements from the given spatial distribution
            instantiation = drawSamples(obj,1,v);
        end
        
        function instantiation = multiBernoulliRFSs(obj,M,r)
            instantiation = cell(M,1);
            for i = 1:M
                %draw an integer v from each Bernoulli distirbution with parameter r
                v = binornd(1,r(i));
                %draw v elements from the given spatial distribution
                instantiation{i} = drawSamples(obj,i,v);
            end
        end
        
        function instantiation = multiBernoulliMixtureRFSs(obj,M,r,p)
            %for simplicity, here we assume that Bernoulli component with
            %the same index in different multi-Bernoulli has the same pdf
            %but probability of existence
            mbidx = mnrnd(1,p)==1;
            %draw an integer v from each Bernoulli distirbution with
            %parameter r of the selected multiBernoulli
            instantiation = multiBernoulliRFSs(obj,M(mbidx),r{mbidx});
        end
        
    end
end

