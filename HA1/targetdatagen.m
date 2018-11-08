function targetdata = targetdatagen(groundtruth,motionmodel)

%generate the tracks
K = groundtruth.K;
targetdata.K = K;
targetdata.X = cell(K,1);
targetdata.N = zeros(K,1);

for i = 1:groundtruth.nbirths
    targetstate = groundtruth.xstart(:,i);
    for k = groundtruth.tbirth(i):min(groundtruth.tdeath(i),K)
        targetstate = motionmodel.A*targetstate;
%         targetstate = motionmodel.A*targetstate + ...
%             mvnrnd(zeros(1,motionmodel.d), motionmodel.Q)';
        targetdata.X{k} = [targetdata.X{k} targetstate];
        targetdata.N(k) = targetdata.N(k) + 1;
    end
end

end


