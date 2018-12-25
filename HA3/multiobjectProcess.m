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
        
        function instance = drawSamples(obj,idxParas,v)
            %draw v elements from the given spatial distribution
            switch obj.spatial_distribution.name
                case 'u'
                    dim = size(obj.spatial_distribution.paras{1}{idxParas},1);
                    instance = zeros(dim,v);
                    for i = 1:dim
                        instance(i,:) = unifrnd(obj.spatial_distribution.paras{1}{idxParas}(i,1),obj.spatial_distribution.paras{1}{idxParas}(i,2),1,v);
                    end
                case 'g'
                    instance = mvnrnd(obj.spatial_distribution.paras{1}{idxParas},obj.spatial_distribution.paras{2}{idxParas},v)';
                case 'gm'
                    gm = gmdistribution(obj.spatial_distribution.paras{1}{idxParas},obj.spatial_distribution.paras{2}{idxParas},obj.spatial_distribution.paras{3}{idxParas});
                    if v == 0
                        instance = [];
                    else
                        instance = random(gm,v)';
                    end
            end
        end
        
        function [instance, card_pmf] = PoissonRFSs(obj,lambda)
            %draw an integer v from Poisson distirbution with parameter
            %lambda
            v = poissrnd(lambda);
            %draw v elements from the given spatial distribution
            instance = drawSamples(obj,1,v);
            card_pmf = @(x) poisspdf(x, lambda);
        end
        
        function [instance, card_pmf] = BernoulliRFSs(obj,r)
            %draw an integer v from Bernoulli distirbution with parameter r
            v = binornd(1,r);
            %draw v elements from the given spatial distribution
            instance = drawSamples(obj,1,v);
            card_pmf = @(x) binopdf(x,1,r);
        end
        
        function [instance, card_pmf] = multiBernoulliRFSs(obj,M,r)
            instance = cell(M,1);
            for i = 1:M
                %draw an integer v from each Bernoulli distirbution with parameter r
                v = binornd(1,r(i));
                %draw v elements from the given spatial distribution
                instance{i} = drawSamples(obj,i,v);
            end
            
            if size(r,1) > 1
                r = r';
            end
            
            lr1 = length(find(r==1));
            r = r(r~=1);
            pcard = [zeros(1,lr1) prod(1-r)*poly(-r./(1-r))];
            card_pmf = @(x) multiobjectProcess.createPMF(pcard,x);
        end
        
        function [instance, card_pmf] = multiBernoulliMixtureRFSs(obj,M,r,p)
            %for simplicity, here we assume that Bernoulli component with
            %the same index in different multi-Bernoulli has the same pdf
            %but probability of existence
            p_norm = p/sum(p);
            mbidx = mnrnd(1,p_norm)==1;
            %draw an integer v from each Bernoulli distirbution with
            %parameter r of the selected multiBernoulli
            instance = multiBernoulliRFSs(obj,M(mbidx),r{mbidx});
            
            num_mb = length(M);
            pcard = zeros(num_mb,max(M)+1);
            for i = 1:num_mb
                lr1 = length(find(r{i}==1));
                r{i} = r{i}(r{i}~=1);
                pcard(i,:) = [zeros(1,lr1) prod(1-r{i})*poly(-r{i}./(1-r{i})) zeros(1,max(M)-M(i))];
            end
            if size(p,1) > 1
                p = p';
            end
            pcard = sum(pcard.*p');
            card_pmf = @(x) multiobjectProcess.createPMF(pcard,x);
        end
        
    end
    
    methods (Static)
        function pmf = createPMF(pcard, pf)
            len = length(pf);
            pmf = zeros(1,len);
            for i = 1:len
                if pf(i)+1 <= length(pcard)
                    pmf(i) = pcard(pf(i)+1);
                end
            end
        end
    end
    
    
end

