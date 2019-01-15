classdef PMBMfilter
    %PMBMFILTER is a class containing necessary functions to implement the PMBM filter
    %DEPENDENCIES: GaussianDensity.m
    %              normalizeLogWeights.m
    %              hypothesisReduction.m
    
    properties
        density %density class handle
        paras   %%parameters specify a PMBM
    end
    
    methods
        function obj = initialize(obj,density_class_handle,birthmodel)
            %INITIATOR initializes PMBMfilter class
            %INPUT: density_class_handle: density class handle
            %       birthmodel: a struct specifying the intensity (mixture) of a PPP birth model
            %OUTPUT:obj.density: density class handle
            %       obj.paras.PPP.w: weights of mixture components in PPP intensity
            %                        --- vector of size (number of mixture components x 1)
            %       obj.paras.PPP.states: parameters of mixture components
            %                             in PPP intensity struct array of size
            %                             (number of mixture components x 1)
            %       obj.MBM.w: weights of MBs--- vector of size (number of MBs (global hypotheses) x 1)
            %       obj.MBM.ht: hypothesis table --- matrix of size (number of global hypotheses x number of tracks)
            %                   entry (h,i) indicates that the (h,i)th single object hypothesis in the ith track
            %                   is included in the hth global hypothesis. If the single object hypothesis is a null
            %                   hypothesis with probability of existence r = 0, (h,i) = 0.
            %       obj.MBM.tt: tracks --- cell of size (number of tracks x 1). The ith cell contains single object
            %       hypotheses in struct form of size (number of single object hypotheses in the ith track x 1).
            %       Each struct has two fields: r: probability of existence; states: parameters specifying the object density
            %       Note that single object hypothesis with r = 0 is not explicitly represented.
            obj.density = density_class_handle;
            obj.paras.PPP.w = log([birthmodel.w]');
            obj.paras.PPP.states = rmfield(birthmodel,'w')';
            obj.paras.MBM.w = [];
            obj.paras.MBM.ht = [];
            obj.paras.MBM.tt = {};
        end
        
        function Bern = Bern_predict(obj,Bern,motionmodel,P_S)
            %BERN_PREDICT performs prediction step for a Bernoulli component
            %INPUT: Bern: a struct that specifies a Bernoulli component,
            %             with fields: r: probability of existence --- scalar;
            %             state: a struct contains parameters describing the object pdf
            %       P_S: object survival probability
            Bern.r = Bern.r*P_S;
            Bern.state = obj.density.predict(Bern.state,motionmodel);
        end
        
        function [Bern, lik_undetected] = Bern_undetected_update(obj,tt_entry,P_D,P_G)
            %BERN_UNDETECTED_UPDATE calculates the likelihood of missed detection,
            %and creates single object hypotheses due to missed detection.
            %INPUT: tt_entry: a (2 x 1) array that specifies the index of single
            %                 object hypotheses. (i,j) indicates the jth
            %                 single object hypothesis in the ith track.
            %       P_D: object detection probability --- scalar
            %       P_G: gating size in probabiliy --- scalar
            %OUTPUT:Bern: a struct that specifies a Bernoulli component,
            %             with fields: r: probability of existence --- scalar;
            %             state: a struct contains parameters describing the object pdf
            %       lik_undetected: missed detection likelihood --- scalar in logorithmic scale
            Bern = obj.paras.MBM.tt{tt_entry(1)}(tt_entry(2));
            l_nodetect = Bern.r*(1 - P_D*P_G);
            lik_undetected = 1 - Bern.r + l_nodetect;
            Bern.r = l_nodetect/lik_undetected;
            lik_undetected = log(lik_undetected);
        end
        
        function lik_detected = Bern_detected_update_lik(obj,tt_entry,z,measmodel,P_D)
            %BERN_DETECTED_UPDATE_LIK calculates the measurement update
            %likelihood for a given single object hypothesis.
            %INPUT: tt_entry: a (2 x 1) array that specifies the index of single
            %                 object hypotheses. (i,j) indicates the jth
            %                 single object hypothesis in the ith track.
            %       z: measurement vector --- (measurement dimension x 1)
            %       P_D: object detection probability --- scalar
            %OUTPUT:lik_detected: measurement update likelihood --- scalar in logarithmic scale
            Bern = obj.paras.MBM.tt{tt_entry(1)}(tt_entry(2));
            lik_detected = obj.density.predictedLikelihood(Bern.state,z,measmodel) + log(P_D*Bern.r);
        end
        
        function Bern = Bern_detected_update_state(obj,tt_entry,z,measmodel)
            %BERN_DETECTED_UPDATE_STATE creates the single object
            %hypothesis due to measurement update.
            %INPUT: tt_entry: a (2 x 1) array that specifies the index of single
            %                 object hypotheses. (i,j) indicates the jth
            %                 single object hypothesis in the ith track.
            %       z: measurement vector --- (measurement dimension x 1)
            %OUTPUT:Bern: a struct that specifies a Bernoulli component,
            %             with fields: r: probability of existence --- scalar;
            %             state: a struct contains parameters describing the object pdf
            Bern = obj.paras.MBM.tt{tt_entry(1)}(tt_entry(2));
            Bern.state = obj.density.update(Bern.state,z,measmodel);
            Bern.r = 1;
        end
        
        function obj = PPP_predict(obj,motionmodel,birthmodel,P_S)
            %PPP_PREDICT performs predicion step for PPP components
            %hypothesising undetected objects.
            %INPUT: P_S: object survival probability --- scalar
            %Predict existing PPP intensity
            obj.paras.PPP.w = obj.paras.PPP.w + log(P_S);
            obj.paras.PPP.states = arrayfun(@(x) obj.density.predict(x,motionmodel), obj.paras.PPP.states);
            %Incorporate PPP birth intensity into PPP intensity
            obj.paras.PPP.w = [obj.paras.PPP.w;log([birthmodel.w]')];
            obj.paras.PPP.states = [obj.paras.PPP.states;rmfield(birthmodel,'w')'];
        end
        
        function [Bern, lik_new] = PPP_detected_update(obj,z,measmodel,P_D,clutter_intensity)
            %PPP_DETECTED_UPDATE creates a single object hypothesis by
            %updating the PPP with measurement and calculates the
            %corresponding likelihood.
            %INPUT: z: measurement vector --- (measurement dimension x 1)
            %       P_D: object detection probability --- scalar
            %       clutter_intensity: Poisson clutter intensity --- scalar
            %OUTPUT:Bern: a struct that specifies a Bernoulli component,
            %             with fields: r: probability of existence --- scalar;
            %             state: a struct contains parameters describing the object pdf
            %       lik_new: measurement update likelihood of PPP --- scalar in logarithmic scale
            state_upd = arrayfun(@(x) obj.density.update(x,z,measmodel), obj.paras.PPP.states);
            w_upd = arrayfun(@(x) obj.density.predictedLikelihood(x,z,measmodel), obj.paras.PPP.states) + obj.paras.PPP.w + log(P_D);
            C = sum(exp(w_upd));
            lik_new = log(C + clutter_intensity);
            Bern.r = C/(C + clutter_intensity);
            w_upd = normalizeLogWeights(w_upd);
            Bern.state = obj.density.momentMatching(w_upd,state_upd);
        end
        
        function obj = PPP_undetected_update(obj,P_D,P_G)
            %PPP_UNDETECTED_UPDATE performs PPP update for missed detection.
            %INPUT: P_D: object detection probability --- scalar
            %       P_G: gating size in probabiliy --- scalar
            obj.paras.PPP.w = obj.paras.PPP.w + log(1 - P_D*P_G);
        end
        
        function obj = PPP_prune(obj,threshold)
            %PPP_PRUNE prunes mixture components in the PPP intensity with small weight.
            %INPUT: threshold: pruning threshold --- scalar in logarithmic scale
            idx = obj.paras.PPP.w > threshold;
            obj.paras.PPP.w = obj.paras.PPP.w(idx);
            obj.paras.PPP.states = obj.paras.PPP.states(idx);
        end
        
        function obj = Bern_prune(obj,prune_threshold)
            %BERN_PRUNE removes Bernoulli components with small probability
            %of existence and re-index the hypothesis table. If a track
            %contains no single object hypothesis after pruning, this track
            %is removed.
            %INPUT: prune_threshold: Bernoulli components with probability
            %of existence smaller than this threshold will be pruned --- scalar
            n_tt = length(obj.paras.MBM.tt);
            for i = 1:n_tt
                %Find all Bernoulli components needed to be pruned
                idx = arrayfun(@(x) x.r<prune_threshold, obj.paras.MBM.tt{i});
                %Prune theses Bernoulli components
                obj.paras.MBM.tt{i} = obj.paras.MBM.tt{i}(~idx);
                idx = find(idx);
                %Update hypothesis table, if a Bernoulli component is
                %pruned, set its corresponding entry to zero
                for j = 1:length(idx)
                    temp = obj.paras.MBM.ht(:,i);
                    temp(temp==idx(j)) = 0;
                    obj.paras.MBM.ht(:,i) = temp;
                end
            end
            
            %Remove tracks that contains only null single object hypotheses
            idx = sum(obj.paras.MBM.ht,1)~=0;
            obj.paras.MBM.ht = obj.paras.MBM.ht(:,idx);
            obj.paras.MBM.tt = obj.paras.MBM.tt(idx);
            if isempty(obj.paras.MBM.ht)
                obj.paras.MBM.w = [];
            end
            
            %Re-index hypothesis table
            n_tt = length(obj.paras.MBM.tt);
            for i = 1:n_tt
                idx = obj.paras.MBM.ht(:,i)==0;
                [~,~,obj.paras.MBM.ht(:,i)] = unique(obj.paras.MBM.ht(:,i),'rows');
                if any(idx)
                    obj.paras.MBM.ht(idx,i) = 0;
                    obj.paras.MBM.ht(~idx,i) = obj.paras.MBM.ht(~idx,i) - 1;
                end
            end
            
            %Merge duplicate hypothesis table rows
            if ~isempty(obj.paras.MBM.ht)
                [ht,~,ic] = unique(obj.paras.MBM.ht,'rows');
                if(size(ht,1)~=size(obj.paras.MBM.ht,1))
                    %There are duplicate entries
                    w = zeros(size(ht,1),1);
                    for i = 1:size(ht,1)
                        indices_dupli = (ic==i);
                        [~,w(i)] = normalizeLogWeights(obj.paras.MBM.w(indices_dupli));
                    end
                    obj.paras.MBM.ht = ht;
                    obj.paras.MBM.w = w;
                end
            end
        end
        
        function obj = Bern_recycle(obj,recycle_threshold,merging_threshold)
            %BERN_RECYCLE recycles Bernoulli components with small
            %probability of existence, add them to the PPP component, and
            %re-index the hypothesis table. If a track contains no single
            %object hypothesis after pruning, this track is removed. After
            %recycling, merge similar Gaussian components in the PPP intensity
            %INPUT: recycle_threshold: Bernoulli components with probability
            %                          of existence smaller than this threshold
            %                          needed to be recycled --- scalar
            %       merging_threshold: merging threshold used in method
            %                          GaussianDensity.mixtureReduction --- scalar
            n_tt = length(obj.paras.MBM.tt);
            for i = 1:n_tt
                idx = arrayfun(@(x) x.r<recycle_threshold, obj.paras.MBM.tt{i});
                %Here, we should also consider the weights of different MBs
                idx_t = find(idx);
                n_h = length(idx_t);
                w_h = zeros(n_h,1);
                for j = 1:n_h
                    idx_h = obj.paras.MBM.ht(:,i) == idx_t(j);
                    [~,w_h(j)] = normalizeLogWeights(obj.paras.MBM.w(idx_h));
                end
                %Recycle
                temp = obj.paras.MBM.tt{i}(idx);
                obj.paras.PPP.w = [obj.paras.PPP.w;log([temp.r]')+w_h];
                obj.paras.PPP.states = [obj.paras.PPP.states;[temp.state]'];
                %Remove Bernoullis
                obj.paras.MBM.tt{i} = obj.paras.MBM.tt{i}(~idx);
                %Update hypothesis table, if a Bernoulli component is
                %pruned, set its corresponding entry to zero
                idx = find(idx);
                for j = 1:length(idx)
                    temp = obj.paras.MBM.ht(:,i);
                    temp(temp==idx(j)) = 0;
                    obj.paras.MBM.ht(:,i) = temp;
                end
            end
            
            %Remove tracks that contains only null single object hypotheses
            idx = sum(obj.paras.MBM.ht,1)~=0;
            obj.paras.MBM.ht = obj.paras.MBM.ht(:,idx);
            obj.paras.MBM.tt = obj.paras.MBM.tt(idx);
            if isempty(obj.paras.MBM.ht)
                obj.paras.MBM.w = [];
            end
            
            %Re-index hypothesis table
            n_tt = length(obj.paras.MBM.tt);
            for i = 1:n_tt
                idx = obj.paras.MBM.ht(:,i)==0;
                [~,~,obj.paras.MBM.ht(:,i)] = unique(obj.paras.MBM.ht(:,i),'rows');
                if any(idx)
                    obj.paras.MBM.ht(idx,i) = 0;
                    obj.paras.MBM.ht(~idx,i) = obj.paras.MBM.ht(~idx,i) - 1;
                end
            end
            
            %Merge duplicate hypothesis table rows
            if ~isempty(obj.paras.MBM.ht)
                [ht,~,ic] = unique(obj.paras.MBM.ht,'rows');
                if(size(ht,1)~=size(obj.paras.MBM.ht,1))
                    %There are duplicate entries
                    w = zeros(size(ht,1),1);
                    for i = 1:size(ht,1)
                        indices_dupli = (ic==i);
                        [~,w(i)] = normalizeLogWeights(obj.paras.MBM.w(indices_dupli));
                    end
                    obj.paras.MBM.ht = ht;
                    obj.paras.MBM.w = w;
                end
            end
            
            %Merge similar Gaussian components in the PPP intensity
            if ~isempty(obj.paras.PPP.w)
                [obj.paras.PPP.w,obj.paras.PPP.states] = obj.density.mixtureReduction(obj.paras.PPP.w,obj.paras.PPP.states,merging_threshold);
            end
        end
        
        function estimates = PMBM_estimator(obj,threshold)
            %PMBM_ESTIMATOR performs object state estimation in the PMBM filter
            %INPUT: threshold (if exist): object states are extracted from Bernoulli
            %components with probability of existence no less than this
            %threhold in Estimator 1. Given the probabilities of detection
            %and survival, this threshold determines the number of consecutive misdetections
            %%%
            %In Estimator 1, we first select the global hypothesis
            %multi-Bernoulli mixture with highest weight. Then, we report
            %the mean of the Bernoulli components in the selected hypothesis
            %whose existence probability is above a threshold.
            %In Estimator 2, we first obtains the MAP estimate of the cardinality n.
            %We can then obtain the highest weight global hypothesis with
            %deterministic cardinality. Once we have found the global hypothesis,
            %the set estimate is formed by the means of the n Bernoulli
            %components with highest existence in this hypothesis.
            
            estimates = [];
            if nargin == 2
                %Estimator 1
                %Find the global hypothesis with the highest weight
                [~,I] = max(obj.paras.MBM.w);
                h_best = obj.paras.MBM.ht(I,:);
                for i = 1:length(h_best)
                    %Check the validity of the single object hypothesis
                    if h_best(i)~=0
                        Bern = obj.paras.MBM.tt{i}(h_best(i));
                        %Report estimates from Bernoulli components
                        %with probability of existence larger than 0.5
                        if Bern.r >= threshold
                            estimates = [estimates obj.density.expectedValue(Bern.state)];
                        end
                    end
                end
            elseif nargin == 1
                %Estimator 2
                num_mb = length(obj.paras.MBM.w);
                r = cell(num_mb,1);
                for i = 1:num_mb
                    ht = obj.paras.MBM.ht(i,:);
                    for j = 1:length(ht)
                        if ht(j)~=0
                            Bern = obj.paras.MBM.tt{j}(ht(j));
                            r{i} = [r{i};Bern.r];
                        end
                    end
                end
                M = cellfun('length',r);
                pcard = zeros(num_mb,max(M)+1);
                %calculate the cardinality pmf for each multi-Bernoulli RFS
                for i = 1:num_mb
                    lr1 = length(find(r{i}==1));
                    temp = r{i}(r{i}~=1);
                    %append zeros to make each multi-Bernoulli's cardinality pmf have equal support
                    pcard(i,:) = [zeros(1,lr1) prod(1-temp)*poly(-temp./(1-temp)) zeros(1,max(M)-M(i))];
                end
                %calculate the cardinality pmf of the multi-Bernoulli mixture
                pcard = sum(pcard.*exp(obj.paras.MBM.w),1);
                [~,I] = max(pcard);
                C_max = I - 1;
                %for each MB hypothesis in the MBM, find the MAP cardinality estimate
                %this step is to remove MB components with unqualified cardinality support
                idx = M >= C_max;
                w = obj.paras.MBM.w(idx);
                r = r(idx);
                ht = obj.paras.MBM.ht(idx,:);
                num_mb = length(w);
                %calculate the weights of global hypotheses with
                %deterministic cardinality n
                w_deter = zeros(num_mb,1);
                for i = 1:num_mb
                    r_sort = sort(r{i},'descend');
                    w_deter(i) = w(i)*prod(r_sort(1:C_max));
                    if length(r{i}) > C_max
                        w_deter(i) = w_deter(i)*prod(1-r_sort(C_max+1:end));
                    end
                end
                %find the highest weight global hypothesis with the
                %deterministic cardinality n
                [~,J] = max(w_deter);
                h_best = ht(J,:);
                r = [];
                for i = 1:length(h_best)
                    if h_best(i)~=0
                        Bern = obj.paras.MBM.tt{i}(h_best(i));
                        r = [r Bern.r];
                        estimates = [estimates obj.density.expectedValue(Bern.state)];
                    end
                end
                %report estimates from the n Bernoulli components with
                %the highest probability of existence
                [~,I] = sort(r,'descend');
                estimates = estimates(:,I(1:C_max));
            end
        end
        
        function obj = PMBM_predict(obj,motionmodel,birthmodel,sensormodel)
            %PMBM_PREDICT performs PMBM prediction step.
            %PPP predict
            obj = PPP_predict(obj,motionmodel,birthmodel,sensormodel.P_S);
            %MBM predict
            for i = 1:length(obj.paras.MBM.tt)
                obj.paras.MBM.tt{i} = arrayfun(@(x) Bern_predict(obj,x,motionmodel,sensormodel.P_S), obj.paras.MBM.tt{i});
            end
        end
        
        function obj = PMBM_update(obj,z,measmodel,sensormodel,gating,wmin,M)
            %PMBM_UPDATE performs PMBM update step.
            %INPUT: z: measurements --- array of size (measurement dimension x number of measurements)
            %       gating: a struct with two fields that specifies gating
            %       parameters: P_G: gating size in decimal --- scalar;
            %       size: gating size --- scalar.
            %       wmin: hypothesis weight pruning threshold --- scalar in logarithmic scale
            %       M: maximum global hypotheses kept
            
            %Update detected objects
            m = size(z,2);                      %number of measurements received
            
            used_meas_u = false(m,1);           %measurement indices inside the gate of undetected objects
            nu = length(obj.paras.PPP.states);  %number of mixture components in PPP intensity
            for i = 1:nu
                %Perform gating for each mixture component in the PPP intensity
                [~,temp] = obj.density.ellipsoidalGating(obj.paras.PPP.states(i),z,measmodel,gating.size);
                used_meas_u = used_meas_u | temp;
            end
            
            n_tt = length(obj.paras.MBM.tt);    %number of pre-existing tracks
            likTable = cell(n_tt,1);            %initialise likelihood table, one for each track
            gating_matrix_d = cell(n_tt,1);
            used_meas_d = false(m,1);           %measurement indices inside the gate of detected objects
            for i = 1:n_tt
                %number of hypotheses in track i
                num_hypo = length(obj.paras.MBM.tt{i});
                %construct gating matrix
                gating_matrix_d{i} = false(m,num_hypo);
                for j = 1:num_hypo
                    %Perform gating for each single object hypothesis
                    Bern_temp = obj.paras.MBM.tt{i}(j);
                    [~,gating_matrix_d{i}(:,j)] = obj.density.ellipsoidalGating(Bern_temp.state,z,measmodel,gating.size);
                end
                used_meas_d = used_meas_d | sum(gating_matrix_d{i},2) >= 1;
            end
            
            %measurement indices inside the gate
            used_meas = used_meas_d | used_meas_u;
            %find indices of measurements inside the gate of detected
            %objects but not undetected objects
            used_meas_d_not_u = used_meas > used_meas_u;
            %find indices of measurements inside the gate of undetected
            %objects but not detected objects
            used_meas_u_not_d = used_meas > used_meas_d;
            %find indices of measurements inside both the gate of detected
            %undetected objects
            used_meas_du = used_meas_d & used_meas_u;
            
            %obtain measurements that are inside the gate of detected objects
            z_d = [z(:,used_meas_du) z(:,used_meas_d_not_u)];
            m = size(z_d,2);
            gating_matrix_d = cellfun(@(x) [x(used_meas_du,:);x(used_meas_d_not_u,:)], gating_matrix_d, 'UniformOutput',false);
            n_tt_upd = n_tt + m;                %number of tracks
            hypoTable = cell(n_tt_upd,1);       %initialise hypothesis table, one for each track
            for i = 1:n_tt
                %number of hypotheses in track i
                num_hypo = length(obj.paras.MBM.tt{i});
                %initialise likelihood table for track i
                likTable{i} = -inf(num_hypo,m+1);
                %initialise hypothesis table for track i
                hypoTable{i} = cell(num_hypo*(m+1),1);
                for j = 1:num_hypo
                    %Missed detection
                    [hypoTable{i}{(j-1)*(m+1)+1},likTable{i}(j,1)] = Bern_undetected_update(obj,[i,j],sensormodel.P_D,gating.P_G);
                    %Update with measurement
                    likTable{i}(j,[false;logical(gating_matrix_d{i}(:,j))]) = ...
                        Bern_detected_update_lik(obj,[i,j],z_d(:,logical(gating_matrix_d{i}(:,j))),measmodel,sensormodel.P_D);
                    for jj = 1:m
                        if gating_matrix_d{i}(jj,j)
                            hypoTable{i}{(j-1)*(m+1)+jj+1} = Bern_detected_update_state(obj,[i,j],z_d(:,jj),measmodel);
                        end
                    end
                end
            end
            
            %%%
            %Update undetected objects
            lik_new = -inf(m,1);
            %Create new tracks, one for each measurement inside the gate
            for i = 1:m
                if i <= length(find(used_meas_du))
                    [hypoTable{n_tt+i,1}{1}, lik_new(i)] = PPP_detected_update(obj,z_d(:,i),measmodel,sensormodel.P_D,sensormodel.lambda_c*sensormodel.pdf_c);
                else
                    %For measurements not inside the gate of undetected
                    %objects, create dummy tracks
                    lik_new(i) = log(sensormodel.lambda_c*sensormodel.pdf_c);
                    hypoTable{n_tt+i,1}{1}.r = 0;
                    hypoTable{n_tt+i,1}{1}.state = [];
                end
            end
            
            %Cost matrix for first detection of undetected objects
            L2 = inf(m);
            L2(logical(eye(m))) = -lik_new;
            
            %Update undetected objects with missed detection
            obj = PPP_undetected_update(obj,sensormodel.P_D,gating.P_G);
            
            %%%
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
                                %do nothing for null single object hypothesis (r = 0)
                                ht_upd(H_upd,i) = 0;
                            else
                                if isempty(idx)
                                    %missed detection
                                    ht_upd(H_upd,i) = (obj.paras.MBM.ht(h,i)-1)*(m+1)+1;
                                else
                                    %measurement update
                                    ht_upd(H_upd,i) = (obj.paras.MBM.ht(h,i)-1)*(m+1)+idx+1;
                                end
                            end
                        end
                        for i = n_tt+1:n_tt_upd
                            idx = find(col4rowBest(:,j)==i, 1);
                            if ~isempty(idx)
                                %measurement update for PPP
                                ht_upd(H_upd,i) = 1;
                            end
                        end
                    end
                end
                
            end
            
            %Append new tracks that created by measurements inside the gate
            %of undetected objects but not detected objects
            z_u_not_d = z(:,used_meas_u_not_d);
            num_u_not_d = size(z_u_not_d,2);
            for i = 1:num_u_not_d
                [hypoTable{n_tt_upd+i,1}{1}, ~] = PPP_detected_update(obj,z_u_not_d(:,i),measmodel,sensormodel.P_D,sensormodel.lambda_c*sensormodel.pdf_c);
            end
            ht_upd = [ht_upd ones(H_upd,num_u_not_d)];
            
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
            [w_upd, hypo_idx] = hypothesisReduction.prune(w_upd,hypo_idx,wmin);
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
            
            %Re-index hypothesis table
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
        
    end
end
