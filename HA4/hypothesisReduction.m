classdef hypothesisReduction
    %HYPOTHESISREDUCTION is class containing different hypotheses reduction method
    %PRUNE:   prune hypotheses with small weights.
    %CAP:     keep M hypotheses with the highest weights and discard the rest.
    %MERGE:   merge similar hypotheses in the sense of small Mahalanobis distance.
    
    methods (Static)
        function [hypothesesWeight_hat, multiHypotheses_hat] = prune(hypothesesWeight, multiHypotheses, threshold)
            %PRUNE prunes hypotheses with small weights
            %INPUT: hypothesesWeight: the weights of different hypotheses in logarithmic scale --- (number of hypotheses) x 1 vector
            %       multiHypotheses: (number of hypotheses) x 1 structure
            %       threshold: hypotheses with weights smaller than this threshold will be discarded --- scalar
            %OUTPUT:hypothesesWeight_hat: hypotheses weights after pruning in logarithmic scale --- (number of hypotheses after pruning) x 1 vector
            %       multiHypotheses_hat: (number of hypotheses after pruning) x 1 structure
            indices_keeped = hypothesesWeight > threshold;
            hypothesesWeight_hat = hypothesesWeight(indices_keeped);
%             [hypothesesWeight_hat,~] = normalizeLogWeights(hypothesesWeight_hat);
            multiHypotheses_hat = multiHypotheses(indices_keeped);
        end
        
        function [hypothesesWeight_hat, multiHypotheses_hat] = cap(hypothesesWeight, multiHypotheses, M)
            %CAP keeps M hypotheses with the highest weights and discard the rest
            %INPUT: hypothesesWeight: the weights of different hypotheses in logarithmic scale --- (number of hypotheses) x 1 vector
            %       multiHypotheses: (number of hypotheses) x 1 structure
            %       M: only keep M hypotheses --- scalar
            %OUTPUT:hypothesesWeight_hat: hypotheses weights after capping in logarithmic scale ---
            %                               (number of hypotheses after capping) x 1 vector
            %       multiHypotheses_hat: (number of hypotheses after capping) x 1 structure
            if length(hypothesesWeight) > M
                [hypothesesWeight, sorted_idx] = sort(hypothesesWeight,'descend');
                hypothesesWeight_hat = hypothesesWeight(1:M);
%                 [hypothesesWeight_hat,~] = normalizeLogWeights(hypothesesWeight_hat);
                multiHypotheses_hat = multiHypotheses(sorted_idx(1:M));
            else
                hypothesesWeight_hat = hypothesesWeight;
                multiHypotheses_hat = multiHypotheses;
            end
        end
        
        function [hypothesesWeight_hat,multiHypotheses_hat] = merge(hypothesesWeight,multiHypotheses,threshold,density)
            %MERGE merges hypotheses within small Mahalanobis distance
            %INPUT: hypothesesWeight: the weights of different hypotheses in logarithmic scale --- (number of hypotheses) x 1 vector
            %       multiHypotheses: (number of hypotheses) x 1 structure
            %       threshold: merging threshold --- scalar
            %       density: a class handle
            %OUTPUT:hypothesesWeight_hat: hypotheses weights after merging in logarithmic scale --- (number of hypotheses after merging) x 1 vector
            %       multiHypotheses_hat: (number of hypotheses after merging) x 1 structure
            
            [hypothesesWeight_hat,multiHypotheses_hat] = density.mixtureReduction(hypothesesWeight,multiHypotheses,threshold);
            
        end
        
        
    end
end