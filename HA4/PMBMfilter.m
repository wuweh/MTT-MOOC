classdef PMBMfilter
    
    properties
        density %density class handle
        paras
    end
    
    methods
        function obj = initialize(obj,density_class_handle,birthmodel)
            obj.density = density_class_handle;
            obj.paras.PPP.w = log([birthmodel.w]');
            obj.paras.PPP.states = rmfield(birthmodel,'w')';
            obj.paras.MBM.w = [];
            obj.paras.MBM.ht = [];
            obj.paras.MBM.tt = cell(0,1);
        end
        
        function obj = Bern_prune(obj,prune_threshold)
            %Remove Bernoulli components with small probability of existence
            n_tt = length(obj.paras.MBM.tt);
            for i = 1:n_tt
                idx = arrayfun(@(x) x.r<prune_threshold, obj.paras.MBM.tt{i});
                obj.paras.MBM.tt{i} = obj.paras.MBM.tt{i}(~idx);
                idx = find(idx);
                for j = 1:length(idx)
                    temp = obj.paras.MBM.ht(:,i);
                    temp(temp==idx(j)) = 0;
                    obj.paras.MBM.ht(:,i) = temp;
                end
            end
            %Remove tracks that contain no single object hypotheses
            idx = sum(obj.paras.MBM.ht,1)~=0;
            obj.paras.MBM.ht = obj.paras.MBM.ht(:,idx);
            obj.paras.MBM.tt = obj.paras.MBM.tt(idx);
            if isempty(obj.paras.MBM.ht)
                obj.paras.MBM.w = [];
            end
            
            %Clean hypothesis table
            n_tt = length(obj.paras.MBM.tt);
            for i = 1:n_tt
                idx = obj.paras.MBM.ht(:,i)==0;
                [~,~,obj.paras.MBM.ht(:,i)] = unique(obj.paras.MBM.ht(:,i),'rows');
                if any(idx)
                    obj.paras.MBM.ht(idx,i) = 0;
                    obj.paras.MBM.ht(~idx,i) = obj.paras.MBM.ht(~idx,i) - 1;
                end
            end
        end
        
        function obj = recycle(obj,recycle_threshold,merging_threshold)
            %Remove Bernoulli components with small probability of existence
            n_tt = length(obj.paras.MBM.tt);
            for i = 1:n_tt
                idx = arrayfun(@(x) x.r<recycle_threshold, obj.paras.MBM.tt{i});
                %Recycle
                temp = obj.paras.MBM.tt{i}(idx);
                obj.paras.PPP.w = [obj.paras.PPP.w;[temp.r]'];
                obj.paras.PPP.states = [obj.paras.PPP.states;[temp.state]'];
                %Remove Bernoullis
                obj.paras.MBM.tt{i} = obj.paras.MBM.tt{i}(~idx);
                
                idx = find(idx);
                for j = 1:length(idx)
                    temp = obj.paras.MBM.ht(:,i);
                    temp(temp==idx(j)) = 0;
                    obj.paras.MBM.ht(:,i) = temp;
                end
                
            end
            %Remove tracks that contain no single object hypotheses
            idx = sum(obj.paras.MBM.ht)~=0;
            obj.paras.MBM.ht = obj.paras.MBM.ht(:,idx);
            obj.paras.MBM.tt = obj.paras.MBM.tt(idx);
            if isempty(obj.paras.MBM.tt)
                obj.paras.MBM.w = [];
            end
            
            n_tt = length(obj.paras.MBM.tt);
            %Clean hypothesis table
            for i = 1:n_tt
                idx = obj.paras.MBM.ht(:,i)==0;
                [~,~,obj.paras.MBM.ht(:,i)] = unique(obj.paras.MBM.ht(:,i),'rows');
                if any(idx)
                    obj.paras.MBM.ht(idx,i) = 0;
                    obj.paras.MBM.ht(~idx,i) = obj.paras.MBM.ht(~idx,i) - 1;
                end
            end
            
            [obj.paras.PPP.w,obj.paras.PPP.states] = obj.density.mixtureReduction(obj.paras.PPP.w,obj.paras.PPP.states,merging_threshold);
        end
        
        function estimates = stateExtraction(obj)
            estimates = [];
            [~,I] = max(obj.paras.MBM.w);
            h_best = obj.paras.MBM.ht(I,:);
            for i = 1:length(h_best)
                if h_best(i)~=0
                    Bern = obj.paras.MBM.tt{i}(h_best(i));
                    if Bern.r >= 0.5
                        estimates = [estimates obj.density.expectedValue(Bern.state)];
                    end
                end
            end
        end
        
        function obj = PMBM_predict(obj,motionmodel,birthmodel,sensormodel)
            obj = PPP_predict(obj,motionmodel,birthmodel,sensormodel.P_S);
            for i = 1:length(obj.paras.MBM.tt)
                obj.paras.MBM.tt{i} = arrayfun(@(x) Bern_predict(obj,x,motionmodel,sensormodel.P_S), obj.paras.MBM.tt{i});
            end
        end
        
        function obj = PMBM_update(obj,z,measmodel,sensormodel,gating,prune_threshold,M)
            %Update detected objects
            m = size(z,2);                      %number of measurements received
            n_tt = length(obj.paras.MBM.tt);    %number of pre-existing tracks
            n_tt_upd = n_tt + m;                %number of tracks
            likTable = cell(n_tt,1);            %initialise likelihood table, one for each track
            hypoTable = cell(n_tt_upd,1);       %initialise hypothesis table, one for each track
            for i = 1:n_tt
                %number of hypotheses in track i
                num_hypo = length(obj.paras.MBM.tt{i});
                %construct gating matrix
                gating_matrix_d = false(m,num_hypo);
                %initialise likelihood table for track i
                likTable{i} = -inf(num_hypo,m+1);
                %initialise hypothesis table for track i
                hypoTable{i} = cell(num_hypo*(m+1),1);
                for j = 1:num_hypo
                    %Perform gating for each single object hypothesis
                    Bern_temp = obj.paras.MBM.tt{i}(j);
                    [~,gating_matrix_d(:,j)] = obj.density.ellipsoidalGating(Bern_temp.state,z,measmodel,gating.size);
                    %Missed detection
                    [hypoTable{i}{(j-1)*(m+1)+1},likTable{i}(j,1)] = Bern_undetected_update(obj,[i,j],sensormodel.P_D,gating.P_G);
                    %Update with measurement
                    likTable{i}(j,[false;logical(gating_matrix_d(:,j))]) = ...
                        Bern_detected_update_lik(obj,[i,j],z(:,logical(gating_matrix_d(:,j))),measmodel,sensormodel.P_D);
                    for jj = 1:m
                        if gating_matrix_d(jj,j)
                            hypoTable{i}{(j-1)*(m+1)+jj+1} = Bern_detected_update_state(obj,[i,j],z(:,jj),measmodel);
                        end
                    end
                end
            end
            
            %Update undetected objects
            lik_new = -inf(m,1);
            %Create new tracks, one for each measurement inside the gate
            for i = 1:m
                [hypoTable{n_tt+i,1}{1}, lik_new(i)] = PPP_detected_update(obj,z(:,i),measmodel,sensormodel.P_D,sensormodel.lambda_c*sensormodel.pdf_c);
            end
            %Cost matrix for first detection of undetected objects
            L2 = inf(m);
            L2(logical(eye(m))) = -lik_new;
            
            %Update undetected objects with missed detection
            obj = PPP_undetected_update(obj,sensormodel.P_D,gating.P_G);
            
            %Update global hypothesis
            w_upd = -inf(0,1);             %initialise global hypothesis weight
            %initialise global hypothesis table, the (h,i)th single object
            %hypothesis in the ith track is included in the hth global
            %hypothesis, (h,i) = 0 means no single object hypothesis is
            %selected in track i.
            ht_upd = zeros(0,n_tt_upd);
            H_upd = 0;
            %number of global hypothesis
            H = length(obj.paras.MBM.w);
            if H == 0
                %if there is no pre-existing track, create new track for each measurement
                w_upd = 0;
                H_upd = 1;
                ht_upd = ones(1,m);
            else
                for h = 1:H
                    %Cost matrix for detected objects
                    L1 = inf(m,n_tt);
                    for i = 1:n_tt
                        hypo_idx = obj.paras.MBM.ht(h,i);
                        if hypo_idx~=0
                            L1(:,i) = -(likTable{i}(hypo_idx,2:end) - likTable{i}(hypo_idx,1));
                        end
                    end
                    %Cost matrix of size m-by-(n+m)
                    L = [L1 L2];
                    %Obtain M best assignments using Murty's algorithm
                    [col4rowBest,~,gainBest]=kBest2DAssign(L,ceil(exp(obj.paras.MBM.w(h))*M));
                    w_upd = [w_upd;-gainBest+obj.paras.MBM.w(h)];
                    %Update global hypothesis look-up table
                    for j = 1:length(gainBest)
                        H_upd = H_upd + 1;
                        for i = 1:n_tt
                            idx = find(col4rowBest(:,j)==i, 1);
                            if obj.paras.MBM.ht(h,i) == 0
                                ht_upd(H_upd,i) = 0;
                            else
                                if isempty(idx)
                                    ht_upd(H_upd,i) = (obj.paras.MBM.ht(h,i)-1)*(m+1)+1;
                                else
                                    ht_upd(H_upd,i) = (obj.paras.MBM.ht(h,i)-1)*(m+1)+idx+1;
                                end
                            end
                        end
                        for i = n_tt+1:n_tt_upd
                            idx = find(col4rowBest(:,j)==i, 1);
                            if ~isempty(idx)
                                ht_upd(H_upd,i) = 1;
                            end
                        end
                    end
                end
            end
            %Remove new created tracks that contain no single object hypotheses
            if ~isempty(ht_upd)
                idx = sum(ht_upd,1)==0;
                ht_upd = ht_upd(:,~idx);
                hypoTable = hypoTable(~idx);
                n_tt_upd = size(ht_upd,2);
            end
            
            %Normalize global hypothesis weights
            w_upd = normalizeLogWeights(w_upd);
            
            %Prune hypotheses with weight smaller than the specified threshold
            hypo_idx = 1:H_upd;
            [w_upd, hypo_idx] = hypothesisReduction.prune(w_upd,hypo_idx,prune_threshold);
            ht_upd = ht_upd(hypo_idx,:);
            w_upd = normalizeLogWeights(w_upd);
            
            %Keep at most M hypotheses with the highest weights
            hypo_idx = 1:length(w_upd);
            [w_upd, hypo_idx] = hypothesisReduction.cap(w_upd,hypo_idx,M);
            ht_upd = ht_upd(hypo_idx,:);
            obj.paras.MBM.w = normalizeLogWeights(w_upd);
            
            %Prune local hypotheses that not appear in maintained global hypotheses
            obj.paras.MBM.tt = cell(n_tt_upd,1);
            for i = 1:n_tt_upd
                temp = ht_upd(:,i);
                hypoTableTemp = hypoTable{i}(unique(temp(temp~=0)));
                obj.paras.MBM.tt{i} = [hypoTableTemp{:}]';
            end
            
            %Clean hypothesis table
            for i = 1:n_tt_upd
                idx = ht_upd(:,i)==0;
                [~,~,ht_upd(:,i)] = unique(ht_upd(:,i),'rows');
                if any(idx)
                    ht_upd(idx,i) = 0;
                    ht_upd(~idx,i) = ht_upd(~idx,i) - 1;
                end
            end
            
            obj.paras.MBM.ht = ht_upd;
        end
        
        %         function Bern = Bern_predict(obj,tt_entry,motionmodel,P_S)
        %             Bern = obj.paras.MBM.tt{tt_entry(1)}(tt_entry(2));
        %             Bern.r = Bern.r*P_S;
        %             Bern.state = obj.density.predict(Bern.state,motionmodel);
        %         end
        
        function Bern = Bern_predict(obj,Bern,motionmodel,P_S)
            Bern.r = Bern.r*P_S;
            Bern.state = obj.density.predict(Bern.state,motionmodel);
        end
        
        function [Bern, lik_undetected] = Bern_undetected_update(obj,tt_entry,P_D,P_G)
            Bern = obj.paras.MBM.tt{tt_entry(1)}(tt_entry(2));
            l_nodetect = Bern.r*(1 - P_D*P_G);
            lik_undetected = 1 - Bern.r + l_nodetect;
            Bern.r = l_nodetect/lik_undetected;
            lik_undetected = log(lik_undetected);
        end
        
        function lik_detected = Bern_detected_update_lik(obj,tt_entry,z,measmodel,P_D)
            Bern = obj.paras.MBM.tt{tt_entry(1)}(tt_entry(2));
            lik_detected = obj.density.predictedLikelihood(Bern.state,z,measmodel) + log(P_D*Bern.r);
        end
        
        function Bern = Bern_detected_update_state(obj,tt_entry,z,measmodel)
            Bern = obj.paras.MBM.tt{tt_entry(1)}(tt_entry(2));
            Bern.state = obj.density.update(Bern.state,z,measmodel);
            Bern.r = 1;
        end
        
        function obj = PPP_predict(obj,motionmodel,birthmodel,P_S)
            obj.paras.PPP.w = obj.paras.PPP.w + log(P_S);
            obj.paras.PPP.states = arrayfun(@(x) obj.density.predict(x,motionmodel), obj.paras.PPP.states);
            obj.paras.PPP.w = [obj.paras.PPP.w;log([birthmodel.w]')];
            obj.paras.PPP.states = [obj.paras.PPP.states;rmfield(birthmodel,'w')'];
        end
        
        function [Bern, lik_new] = PPP_detected_update(obj,z,measmodel,P_D,clutter_intensity)
            state_upd = arrayfun(@(x) obj.density.update(x,z,measmodel), obj.paras.PPP.states);
            w_upd = arrayfun(@(x) obj.density.predictedLikelihood(x,z,measmodel), obj.paras.PPP.states) + obj.paras.PPP.w + log(P_D);
            C = sum(exp(w_upd));
            lik_new = log(C + clutter_intensity);
            Bern.r = C/(C + clutter_intensity);
            w_upd = normalizeLogWeights(w_upd);
            Bern.state = obj.density.momentMatching(w_upd,state_upd);
        end
        
        function obj = PPP_undetected_update(obj,P_D,P_G)
            obj.paras.PPP.w = obj.paras.PPP.w + log(1 - P_D*P_G);
        end
        
        function obj = PPP_prune(obj,threshold)
            idx = obj.paras.PPP.w > threshold;
            obj.paras.PPP.w = obj.paras.PPP.w(idx);
            obj.paras.PPP.states = obj.paras.PPP.states(idx);
        end
        
        
    end
end

