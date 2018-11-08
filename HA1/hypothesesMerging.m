function [hypothesesWeightMerged,multiHypothesesMerged]= hypothesesMerging(hypothesesWeight,multiHypotheses,threshold)
%HYPOTHESESMERGING merges hypotheses within small Mahalanobis distance
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
I = 1:length(multiHypotheses);
el = 1;

multiHypothesesMerged = struct('x',0,'P',0);

while ~isempty(I)
    Ij = [];
    [~,j] = max(hypothesesWeight);
    iPt = inv(multiHypotheses(j).P);
    for i = I
        temp = multiHypotheses(i).x-multiHypotheses(j).x;
        val = temp'*iPt*temp;
        if val <= threshold
            Ij= [ Ij i ];
        end
    end
    
    %Merge hypotheses within small Mahalanobis distance
    hypothesesWeightMerged(el,1) = sum(hypothesesWeight(Ij));
    multiHypothesesMerged(el).x = wsumvec(hypothesesWeight(Ij),[multiHypotheses(Ij).x],s_d)/hypothesesWeightMerged(el,1);
    multiHypothesesMerged(el).P = wsummat(hypothesesWeight(Ij),reshape([multiHypotheses(Ij).P],[s_d,s_d,length(Ij)]),s_d)/hypothesesWeightMerged(el,1);
    
    I = setdiff(I,Ij);
    hypothesesWeight(Ij,1) = -1;
    el = el+1;
end

%Normalize the weights
hypothesesWeightMerged = hypothesesWeightMerged/sum(hypothesesWeightMerged);

end

function out = wsumvec(w,vecstack,xdim)
    wmat = repmat(w',[xdim,1]);
    out  = sum(wmat.*vecstack,2);
end

function out = wsummat(w,matstack,xdim)
    w = reshape(w,[1,1,size(w)]);
    wmat = repmat(w,[xdim,xdim,1]);
    out = sum(wmat.*matstack,3);
end
