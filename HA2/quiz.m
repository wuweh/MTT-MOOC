clc;clear

load('pos2.mat');
figure
hold all
grid on
h1 = plot(xlog{1}(:,1),xlog{1}(:,2),'linewidth',2);
h2 = plot(xlog{2}(:,1),xlog{2}(:,2),'linewidth',2);
h3 = plot(xlog{3}(:,1),xlog{3}(:,2),'linewidth',2);
xlim([-2000 2000])
ylim([-2000 3500])
xlabel('x')
ylabel('y')

h4 = plot(xlog{1}(1,1),xlog{1}(1,2),'ro','linewidth',2,'MarkerSize',10);
plot(xlog{2}(1,1),xlog{2}(1,2),'ro','linewidth',2,'MarkerSize',10)
plot(xlog{3}(1,1),xlog{3}(1,2),'ro','linewidth',2,'MarkerSize',10)


h5 = plot(xlog{1}(end,1),xlog{1}(end,2),'rx','linewidth',2,'MarkerSize',10);
plot(xlog{2}(end,1),xlog{2}(end,2),'rx','linewidth',2,'MarkerSize',10)
plot(xlog{3}(end,1),xlog{3}(end,2),'rx','linewidth',2,'MarkerSize',10)

legend([h1 h2 h3 h4 h5],'Object 1','Object 2','Object 3','Start position','End position','location','best')
title('Ground Truth')

%%
%Choose object detection probability
P_D = 0.999;
%Choose clutter rate
lambda_c = 0.001;

%Creat sensor model
range_c = [-2000 2000;0 2000];
sensor_model = modelgen.sensormodel(P_D,lambda_c,range_c);

%Creat linear motion model
T = 1;
sigma_q = 10;
motion_model = motionmodel.cvmodel(T,sigma_q);

%Create linear measurement model
sigma_r = 5;
meas_model = measmodel.cvmeasmodel(sigma_r);

%Creat ground truth model
nbirths = 3;
K = 51;
tbirth = zeros(nbirths,1);
tdeath = zeros(nbirths,1);

initial_state = repmat(struct('x',[],'P',eye(motion_model.d)),[1,nbirths]);

initial_state(1).x = [xlog{1}(1,1); xlog{1}(1,2); xlog{1}(2,1)-xlog{1}(1,1); 0];   tbirth(1) = 1;   tdeath(1) = K;
initial_state(2).x = [xlog{2}(1,1); xlog{2}(1,2); xlog{2}(2,1)-xlog{2}(1,1); 0];   tbirth(2) = 1;   tdeath(2) = K;
initial_state(3).x = [xlog{3}(1,1); xlog{3}(1,2); 0; xlog{3}(2,2)-xlog{3}(1,2)];   tbirth(3) = 1;   tdeath(3) = K;

%% Generate true object data (noisy or noiseless) and measurement data
groundtruth = modelgen.groundtruth(nbirths,[initial_state.x],tbirth,tdeath,K);
ifnoisy = 0;
% objectdata = objectdatagen(ground_truth,motion_model,ifnoisy);

%Generate the tracks
K = groundtruth.K;
objectdata.X = cell(K,1);
objectdata.N = zeros(K,1);

for i = 1:groundtruth.nbirths
    objectstate = groundtruth.xstart(:,i);
    for k = groundtruth.tbirth(i):min(groundtruth.tdeath(i),K)
        objectstate = [xlog{i}(k,1); xlog{i}(k,2); 0; 0];
        objectdata.X{k} = [objectdata.X{k} objectstate];
        objectdata.N(k) = objectdata.N(k) + 1;
    end
end


%% Single object tracker parameter setting
P_G = 0.999;            %gating size in percentage
wmin = 1e-6;            %hypothesis pruning threshold
merging_threshold = 4;  %hypothesis merging threshold
M = 100;                %maximum number of hypotheses kept in MHT
density_class_handle = @GaussianDensity;    %density class handle
tracker = n_objectracker();
tracker = tracker.initialize(density_class_handle,P_G,meas_model.d,wmin,merging_threshold,M);
    
measdata = measdatagen(objectdata,sensor_model,meas_model);

% measdata{17}(:,2) = [];
% measdata{18}(:,2) = [];
% measdata{19}(:,2) = [];
% measdata{20}(:,2) = [];
% measdata{21}(:,2) = [];

% sensor_model.P_D = 0.9;

% NN tracker
GNNestimates = GNNtracker(tracker, initial_state, measdata, sensor_model, motion_model, meas_model);

% JPDA tracker
JPDAestimates = JPDAtracker(tracker, initial_state, measdata, sensor_model, motion_model, meas_model);

% Multi-hypothesis tracker
TOMHTestimates = TOMHT(tracker, initial_state, measdata, sensor_model, motion_model, meas_model);


%% Ploting
figure
hold on

meas = cell2mat(measdata');

true_state = cell2mat(objectdata.X');
GNN_estimated_state = cell2mat(GNNestimates');

h1 = plot(true_state(1,:), true_state(2,:), 'g*');
h2 = plot(meas(1,:), meas(2,:),'kx');
h3 = plot(GNN_estimated_state(1,:), GNN_estimated_state(2,:),'ro');


legend([h1 h2 h3],'Ground Truth','Measurements','GNN', 'Location', 'best')
%%

figure
hold on

JPDA_estimated_state = cell2mat(JPDAestimates');
h1 = plot(true_state(1,:), true_state(2,:), 'g*');
h2 = plot(meas(1,:), meas(2,:),'kx');
h3 = plot(JPDA_estimated_state(1,:), JPDA_estimated_state(2,:),'mo');
legend([h1 h2 h3],'Ground Truth','Measurements','JPDA estimates', 'Location', 'best')
xlim([-2000 2000])
ylim([-2000 3500])
xlabel('x')
ylabel('y')

title('JPDA Estimates v.s. Ground Truth')
%%
figure
hold on

TOMHT_estimated_state = cell2mat(TOMHTestimates');
h1 = plot(true_state(1,:), true_state(2,:), 'g*');
h2 = plot(meas(1,:), meas(2,:),'kx');
h3 = plot(TOMHT_estimated_state(1,:), TOMHT_estimated_state(2,:),'mo');
legend([h1 h2 h3],'Ground Truth','Measurements','TOMHT', 'Location', 'best')