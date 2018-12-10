function [assignments,costs]= assign2DByGibbs(L,numIteration,M)

n = size(L,1);
m = size(L,2) - n;

assignments= zeros(numIteration,n);
costs= zeros(numIteration,1);

currsoln= m+1:m+n; %use all missed detections as initial solution
assignments(1,:)= currsoln;
costs(1)=sum(L(sub2ind(size(L),1:n,currsoln)));
for sol= 2:numIteration
    for var= 1:n
        tempsamp= exp(-L(var,:)); %grab row of costs for current association variable
        tempsamp(currsoln([1:var-1,var+1:end]))= 0; %lock out current and previous iteration step assignments except for the one in question
        idxold= find(tempsamp>0); tempsamp= tempsamp(idxold);
        [~,currsoln(var)]= histc(rand(1,1),[0;cumsum(tempsamp(:))/sum(tempsamp)]);
        currsoln(var)= idxold(currsoln(var));
    end
    assignments(sol,:)= currsoln;
    costs(sol)= sum(L(sub2ind(size(L),1:n,currsoln)));
end
[C,I,~]= unique(assignments,'rows');
assignments= C;
costs= costs(I);

if length(costs) > M
    [costs, sorted_idx] = sort(costs);
    costs = costs(1:M);
    assignments = assignments(sorted_idx(1:M),:);
end

assignments = assignments';

end