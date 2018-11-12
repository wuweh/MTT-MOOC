classdef hypothesisreduction < handle
    %HYPOTHESISREDUCTION is class containing different hypotheses reduction
    %method
    %PRUNE: prune hypotheses with small weights.
    %CAP: keep M hypotheses with the highest weights and discard the rest.
    %MERGE: merge hypotheses within small Mahalanobis distance.
    
    methods (Static)
        function [hypothesesWeight_update, multiHypotheses_update] = ...
                prune(hypothesesWeight, multiHypotheses, threshold)
            %PRUNE prunes hypotheses with small weights
            %INPUT: hypothesesWeight: the weights of different hypotheses ---
            %                       (number of hypotheses) x 1 vector
            %       multiHypotheses: (number of hypotheses) x 1 structure
            %                       with two fields: x: target state mean;
            %                       P: target state covariance
            %       threshold: hypotheses with weights smaller than this threshold
            %                   will be discarded --- scalar
            %OUTPUT:hypothesesWeight_update: hypotheses weights after pruning---
            %                               (number of hypotheses after pruning) x 1 vector
            %       multiHypotheses_update: (number of hypotheses after pruning) x 1 structure
            indices_keeped = hypothesesWeight > threshold;
            hypothesesWeight_update = hypothesesWeight(indices_keeped);
            hypothesesWeight_update = hypothesesWeight_update/sum(hypothesesWeight_update);
            multiHypotheses_update = multiHypotheses(indices_keeped);
        end
        
        function [hypothesesWeight_update, multiHypotheses_update] = ...
                cap(hypothesesWeight, multiHypotheses, M)
            %CAP keeps M hypotheses with the highest weights and discard
            %the rest
            %INPUT: hypothesesWeight: the weights of different hypotheses ---
            %                       (number of hypotheses) x 1 vector
            %       multiHypotheses: (number of hypotheses) x 1 structure
            %                       with two fields: x: target state mean;
            %                       P: target state covariance
            %       M: only keep M hypotheses --- scalar
            %OUTPUT:hypothesesWeight_update: hypotheses weights after capping---
            %                               (number of hypotheses after capping) x 1 vector
            %       multiHypotheses_update: (number of hypotheses after capping) x 1 structure
            if length(hypothesesWeight) > M
                [hypothesesWeight, sorted_idx] = sort(hypothesesWeight,'descend');
                hypothesesWeight_update = hypothesesWeight(1:M);
                hypothesesWeight_update = hypothesesWeight_update/sum(hypothesesWeight_update);
                multiHypotheses_update = multiHypotheses(sorted_idx(1:M));
            else
                hypothesesWeight_update = hypothesesWeight;
                multiHypotheses_update = multiHypotheses;
            end
        end
        
        function [hypothesesWeightMerged,multiHypothesesMerged]= merge(hypothesesWeight,multiHypotheses,threshold)
            %MERGE merges hypotheses within small Mahalanobis distance
            %INPUT: hypothesesWeight: the weights of different hypotheses ---
            %                       (number of hypotheses) x 1 vector
            %       multiHypotheses: (number of hypotheses) x 1 structure
            %                       with two fields: x: target state mean;
            %                       P: target state covariance
            %       threshold: hypotheses with Mahalanobis distance smaller this
            %                   value will be merged --- scalar
            %OUTPUT:hypothesesWeightMerged: hypotheses weights after merging---
            %                               (number of hypotheses after merging) x 1 vector
            %       multiHypothesesMerged: (number of hypotheses after merging) x 1 structure
            
            s_d = size(multiHypotheses(1).x,1);
            %Index set of hypotheses
            I = 1:length(multiHypotheses);
            el = 1;
            
            multiHypothesesMerged = struct('x',0,'P',0);
            
            while ~isempty(I)
                Ij = [];
                %Find the hypothesis with the highest weight
                [~,j] = max(hypothesesWeight);
                iPt = inv(multiHypotheses(j).P);
                for i = I
                    temp = multiHypotheses(i).x-multiHypotheses(j).x;
                    val = temp'*iPt*temp;
                    %Find other similar hypotheses in the sense of small Mahalanobis distance
                    if val <= threshold
                        Ij= [ Ij i ];
                    end
                end
                
                %Merge hypotheses (weighted average) within small Mahalanobis distance
                hypothesesWeightMerged(el,1) = sum(hypothesesWeight(Ij));
                multiHypothesesMerged(el).x = wsumvec(hypothesesWeight(Ij),[multiHypotheses(Ij).x],s_d)/hypothesesWeightMerged(el,1);
                multiHypothesesMerged(el).P = wsummat(hypothesesWeight(Ij),reshape([multiHypotheses(Ij).P],[s_d,s_d,length(Ij)]),s_d)/hypothesesWeightMerged(el,1);
                
                %Remove indices of merged hypotheses from hypotheses index set
                I = setdiff(I,Ij);
                %Set a negative to make sure this hypothesis won't be selected again
                hypothesesWeight(Ij,1) = -1;
                el = el+1;
            end
            
            %Normalize the weights
            hypothesesWeightMerged = hypothesesWeightMerged/sum(hypothesesWeightMerged);
            
            function out = wsumvec(w,vecstack,xdim)
                wmat = repmat(w',[xdim,1]);
                out  = sum(wmat.*vecstack,2);
            end
            
            function out = wsummat(w,matstack,xdim)
                w = reshape(w,[1,1,size(w)]);
                wmat = repmat(w,[xdim,xdim,1]);
                out = sum(wmat.*matstack,3);
            end
            
        end
        
    end
end

