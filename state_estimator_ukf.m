function state_estimator_ukf
close all;
clear all;
freq=10000;% in hertz keep it as high as possible - paper has done it on very slow moving robot
T=60;
data_T=200;
ctrl_freq=400;
offset_r=[0;0;0];
assumeFlatFloor=true;
only_predict=false;

%% get the data
load('ground_truth.mat');
ctrl_state=load('ctrl_state.mat');
ctrl_state=ctrl_state.ctrl_state;
noise_cov=setNoiseCov();
obs_sim=load('obs_sim.mat');
obs_sim=obs_sim.obs_sim;
[gt_mean,obs_sim]=modifyDataFreq(gt_mean,obs_sim, freq);
angles=[obs_sim(1,1:T*freq+1);obs_sim(8:end,1:T*freq+1)];
acc_sensor_w=obs_sim(1:4,1:T*freq+1);
gyro_sensor_w=[obs_sim(1,1:T*freq+1);obs_sim(5:7,1:T*freq+1)];
if (freq<=ctrl_freq)
    ctrl_state=ctrl_state(:,1:ctrl_freq/freq:end);
    ctrl_state=ctrl_state(:,1:T*freq+1);
else
    ctrl_state_exterp(1,:)=gt_mean(1,:);
    ctrl_state_exterp(2,:)=interp1(ctrl_state(1,:),ctrl_state(2,:),ctrl_state_exterp(1,:));
    ctrl_state=ctrl_state_exterp;
    ctrl_state=ctrl_state(:,1:T*freq+1);
end
%% set initial variables
time_gt=gt_mean(1,1:T*freq+1);
dt=gt_mean(1,2)-gt_mean(1,1);
r_gt_w=gt_mean(2:4,1:T*freq+1);
v_gt_w=gt_mean(5:7,1:T*freq+1);
q_gt_b=gt_mean(8:11,1:T*freq+1);
[roll_gt,pitch_gt,yaw_gt]=quat2rpy(q_gt_b);
p_fl_gt_w=gt_mean(12:14,1:T*freq+1);
p_fr_gt_w=gt_mean(15:17,1:T*freq+1);
p_rl_gt_w=gt_mean(18:20,1:T*freq+1);
p_rr_gt_w=gt_mean(21:23,1:T*freq+1);
feet_gt=[p_fl_gt_w;p_fr_gt_w;p_rl_gt_w;p_rr_gt_w];

    function predict_state()
    end
    
    function encUpdateState()
    end
    
    function outlierDetection()
    end
end