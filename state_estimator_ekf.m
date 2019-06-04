function state_estimator_ekf
% init_quad_new_leg_raibert_strategy_Naman
close all;
clear all;
freq=10000;% in hertz keep it as high as possible - paper has done it on very slow moving robot
T=10;
data_T=200;
ctrl_freq=400;
offset_r=[0;0;0];
assumeFlatFloor=true;
only_predict=true;
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

%% set initial variables

%robot body size
body.x_length = 0.6;
body.y_length = 0.3;
body.z_length = 0.15;
body.shoulder_size = 0.07;  
body.upper_length = 0.16;
body.lower_length = 0.25;
body.foot_radius = 0.035;
body.shoulder_distance = 0.2;
body.max_stretch = body.upper_length + body.lower_length;
body.knee_damping = 0.1;


num_states=28;
bias_acc=zeros(3,1);
bias_gyro=zeros(3,1);

t=1;
r_upd_prev_w(:,t)=r_gt_w(:,t)+offset_r; %m
v_upd_prev_w(:,t)=v_gt_w(:,t);
q_upd_prev_b(:,t)=q_gt_b(:,t);%[w;x;y;z]
p_fl_upd_prev_w(:,t)=p_fl_gt_w(:,t);
p_fr_upd_prev_w(:,t)=p_fr_gt_w(:,t);
p_rl_upd_prev_w(:,t)=p_rl_gt_w(:,t);
p_rr_upd_prev_w(:,t)=p_rr_gt_w(:,t);

pcov_upd_prev=setInitCov(num_states);

R_bw_upd_prev=quat2rotm(q_upd_prev_b');
R_wb_upd_prev=R_bw_upd_prev';

contact_legs=zeros(4,1);
feet_lin=zeros(4,3);
%% filter
r_pred_new_w=[];
v_pred_new_w=[];
q_pred_new_b=[];
p_fl_pred_new_w=[];
p_fr_pred_new_w=[];
p_rl_pred_new_w=[];
p_rr_pred_new_w=[];

for t=1:T*freq
    R_bw_gt=quat2rotm(q_gt_b(:,t)');
    acc_corrected=R_bw_gt*acc_sensor_w(2:4,t)-bias_acc;
    gyro_corrected=R_bw_gt*gyro_sensor_w(2:4,t)-bias_gyro;
    %% prediction
    r_pred_new_w(:,t)=r_upd_prev_w(:,t)+dt*v_upd_prev_w(:,t)+((dt^2)/2)*(R_wb_upd_prev*acc_corrected);
    v_pred_new_w(:,t)=v_upd_prev_w(:,t)+dt*(R_wb_upd_prev*acc_corrected);
    angle=norm(dt*gyro_corrected);
    axis=(dt*gyro_corrected)/angle;
    if(angle<1e-10)
        q_pred_new_b(:,t)=q_upd_prev_b(:,t);
    else
        q_pred_new_b(:,t)=quatmultiply(axang2quat([axis' angle]),q_upd_prev_b(:,t)')';
    end
    R_bw_pred_new=quat2rotm(q_pred_new_b(:,t)');
    R_wb_pred_new=R_bw_pred_new';
    p_fl_pred_new_w(:,t)=p_fl_upd_prev_w(:,t);
    p_fr_pred_new_w(:,t)=p_fr_upd_prev_w(:,t);
    p_rl_pred_new_w(:,t)=p_rl_upd_prev_w(:,t);
    p_rr_pred_new_w(:,t)=p_rr_upd_prev_w(:,t);
    bias_acc_pred_b=bias_acc;
    bias_gyro_pred_b=bias_gyro;
    pred_state=[r_pred_new_w(:,t)' v_pred_new_w(:,t)' q_pred_new_b(:,t)' p_fl_pred_new_w(:,t)' p_fr_pred_new_w(:,t)'...
        p_rl_pred_new_w(:,t)' p_rr_pred_new_w(:,t)' bias_acc_pred_b' bias_gyro_pred_b']';
    if(t>1)
        axang_vel_b= quat2axang(quatmultiply(q_pred_new_b(:,t)',quatinv(q_pred_new_b(:,t-1)')));
        w_lin=1/dt*(axang_vel_b(end)*axang_vel_b(1:end-1));
        R_bw_last=quat2rotm(q_pred_new_b(:,t-1)');
        f1_lin=R_bw_last*((r_pred_new_w(:,t)-r_pred_new_w(:,t-1)-dt*v_pred_new_w(:,t-1))/(0.5*dt^2));
        f2_lin=R_bw_last*(v_pred_new_w(:,t)-v_pred_new_w(:,t-1))/dt;
    else
        w_lin=gyro_corrected;
        f1_lin=acc_corrected;
        f2_lin=acc_corrected;
    end   
    % equation 44-45
    F_k=error_dynamics_matrix(dt,f1_lin, f2_lin, w_lin,R_wb_upd_prev);
    pred_cov=F_k*pcov_upd_prev*F_k'+process_noise_cov(dt, noise_cov, R_wb_upd_prev, w_lin);
    pred_cov_diag=diag(pred_cov);
    %% update
    if(only_predict)
        r_upd_prev_w(:,t+1)=r_pred_new_w(:,t);
        v_upd_prev_w(:,t+1)=v_pred_new_w(:,t);
        q_upd_prev_b(:,t+1)=q_pred_new_b(:,t);
        R_bw_upd_prev=quat2rotm(q_upd_prev_b(:,t+1)');
        R_wb_upd_prev=R_bw_upd_prev';
        p_fl_upd_prev_w(:,t+1)=p_fl_upd_prev_w(:,t);
        p_fr_upd_prev_w(:,t+1)=p_fr_upd_prev_w(:,t);
        p_rl_upd_prev_w(:,t+1)=p_rl_upd_prev_w(:,t);
        p_rr_upd_prev_w(:,t+1)=p_rr_upd_prev_w(:,t);
    else
%         measured_angles=angles(2:end,t);%+repmat(sqrt(diag(noise_cov.R_alpha)).*(-1+2*rand([3,1])),4,1);
%         measured_s=measured_kin(measured_angles,body, noise_cov);
        R_bw_gt=quat2rotm(q_gt_b(:,t)');
        measured_s=[R_bw_gt*(p_fl_gt_w(:,t)-r_gt_w(:,t));...
                    R_bw_gt*(p_fr_gt_w(:,t)-r_gt_w(:,t));...
                    R_bw_gt*(p_rl_gt_w(:,t)-r_gt_w(:,t));...
                    R_bw_gt*(p_rr_gt_w(:,t)-r_gt_w(:,t))];
%         measured_s=measured_s+sqrt(1e-4).*(-1+2*rand(12,1));
        %% measured_angles
        fl_offset = [body.shoulder_distance;body.y_length/2+body.shoulder_size/2;0];
        fr_offset = [body.shoulder_distance;-body.y_length/2-body.shoulder_size/2;0];
        rl_offset = [-body.shoulder_distance;body.y_length/2+body.shoulder_size/2;0];
        rr_offset = [-body.shoulder_distance;-body.y_length/2-body.shoulder_size/2;0];
        [measured_angles(1),measured_angles(2),measured_angles(3)] = inverse_kinematics(measured_s(1:3)-fl_offset, body);
        [measured_angles(4),measured_angles(5),measured_angles(6)] = inverse_kinematics(measured_s(4:6)-fr_offset, body);
        [measured_angles(7),measured_angles(8),measured_angles(9)] = inverse_kinematics(measured_s(7:9)-rl_offset, body);
        [measured_angles(10),measured_angles(11),measured_angles(12)] = inverse_kinematics(measured_s(10:12)-rr_offset, body);
%         measured_s=measured_kin(measured_angles,body, noise_cov);
%         error=measured_s-measured_s_new;
%         norm(error)
        if (t>1)
            if(ctrl_state(2,t)==2)
                if(ctrl_state(2,t-1)==2)
                    contact_legs=contact_legs+[1;0;0;1];
                else
                    contact_legs=[1;0;0;1];
                end
            elseif (ctrl_state(2,t)==3)
                if(ctrl_state(2,t-1)==3)
                    contact_legs=contact_legs+[0;1;1;0];
                else
                    contact_legs=[0;1;1;0];
                end
            elseif (ctrl_state(2,t)==1)
                if(ctrl_state(2,t-1)==1)
                    contact_legs=contact_legs+ones(4,1);
                else
                    contact_legs=ones(4,1);
                end
            else
                r_upd_prev_w(:,t+1)=r_pred_new_w(:,t);
                v_upd_prev_w(:,t+1)=v_pred_new_w(:,t);
                q_upd_prev_b(:,t+1)=q_pred_new_b(:,t);
                R_bw_upd_prev=quat2rotm(q_upd_prev_b(:,t+1)');
                R_wb_upd_prev=R_bw_upd_prev';
                p_fl_upd_prev_w(:,t+1)=p_fl_gt_w(:,t);
                p_fr_upd_prev_w(:,t+1)=p_fr_gt_w(:,t);
                p_rl_upd_prev_w(:,t+1)=p_rl_gt_w(:,t);
                p_rr_upd_prev_w(:,t+1)=p_rr_gt_w(:,t);
                continue;
            end
        else
            r_upd_prev_w(:,t+1)=r_pred_new_w(:,t);
            v_upd_prev_w(:,t+1)=v_pred_new_w(:,t);
            q_upd_prev_b(:,t+1)=q_pred_new_b(:,t);
            R_bw_upd_prev=quat2rotm(q_upd_prev_b(:,t+1)');
            R_wb_upd_prev=R_bw_upd_prev';
            p_fl_upd_prev_w(:,t+1)=p_fl_gt_w(:,t);
            p_fr_upd_prev_w(:,t+1)=p_fr_gt_w(:,t);
            p_rl_upd_prev_w(:,t+1)=p_rl_gt_w(:,t);
            p_rr_upd_prev_w(:,t+1)=p_rr_gt_w(:,t);
            continue;
        end
%         contact_legs=ones(4,1);
        num_contacts=sum(contact_legs>=1);
        
        % Handle initialization of new contacts
        for i=1:4
            if(contact_legs(i)==1)
                pred_state(10+3*(i-1)+1:10+3*i)=r_pred_new_w(:,t)+R_wb_pred_new*measured_s(3*(i-1)+1:3*i);
                if (assumeFlatFloor)
                    pred_state(10+3*i)=0.0327;
                end
                pcov_upd_prev(:,9+3*(i-1)+1:9+3*i)=zeros();
                pcov_upd_prev(9+3*(i-1)+1:9+3*i,:)=zeros();
                pcov_upd_prev(9+3*(i-1)+1:9+3*i,9+3*(i-1)+1:9+3*i)=eye(3)*1e-10;
            end
        end
%         
        for i=1:4
            if(contact_legs(i)==1)
                feet_lin(i,:)=pred_state(10+3*(i-1)+1:10+3*i);
            end
        end
        
        measurement_residual=zeros(3*num_contacts,1);
        H_k=zeros(3*num_contacts, 9+3*4+6);
        R_k=zeros(3*num_contacts,3*num_contacts);
        
        if(num_contacts>0)
            % equation 25 and 48
            R_feet=zeros(4,3,3);
            Jac_fl=computeJacobian(measured_angles(1),measured_angles(2),measured_angles(3),body);
            R_feet(1,:,:)=noise_cov.R_s+Jac_fl*noise_cov.R_alpha*Jac_fl';

            Jac_fr=computeJacobian(measured_angles(4),measured_angles(5),measured_angles(6),body);
            R_feet(2,:,:)=noise_cov.R_s+Jac_fr*noise_cov.R_alpha*Jac_fr';

            Jac_rl=computeJacobian(measured_angles(7),measured_angles(8),measured_angles(9),body);
            R_feet(3,:,:)=noise_cov.R_s+Jac_rl*noise_cov.R_alpha*Jac_rl';

            Jac_rr=computeJacobian(measured_angles(10),measured_angles(11),measured_angles(12),body);
            R_feet(4,:,:)=noise_cov.R_s+Jac_rr*noise_cov.R_alpha*Jac_rr';

            j=1;
            for i=1:4
                if (contact_legs(i)>=1) 
                    measurement_residual(3*(j-1)+1:3*j,1)=measured_s(3*(i-1)+1:3*i,1)-R_bw_pred_new*(pred_state(10+3*(i-1)+1:10+3*i)-r_pred_new_w(:,t));
                    H_k(3*(j-1)+1:3*j,1:3)=-1*R_bw_pred_new;
                    H_k(3*(j-1)+1:3*j,7:9)=R_bw_pred_new*skewsym_op(feet_lin(i,:)'-r_pred_new_w(:,t))*R_wb_pred_new;
                    H_k(3*(j-1)+1:3*j,9+3*(j-1)+1:9+3*j)=R_bw_pred_new;
                    R_k(3*(j-1)+1:3*j,3*(j-1)+1:3*j)=R_feet(i,:,:);
                    j=j+1;
                end
            end
        end
        
        S_k=H_k*pred_cov*H_k'+R_k;
        K_k=pred_cov*H_k'*inv(S_k);
        delta_x=K_k*measurement_residual;%27x1
        upd_state_cov=(eye(27)-K_k*H_k)*pred_cov;
        upd_cov_diag=diag(upd_state_cov);
        % equation 53
        r_upd_prev_w(:,t+1)=pred_state(1:3)+delta_x(1:3);
        v_upd_prev_w(:,t+1)=pred_state(4:6)+delta_x(4:6);
        delta_angle=norm(delta_x(7:9)); 
        delta_axis=delta_x(7:9)/delta_angle;
        if delta_angle<1e-10
            q_upd_prev_b(:,t+1)=pred_state(7:10);
        else
            q_upd_prev_b(:,t+1)=quatmultiply(axang2quat([delta_axis' delta_angle]),pred_state(7:10)')';
        end
        R_bw_upd_prev=quat2rotm(q_upd_prev_b(:,t+1)');
        R_wb_upd_prev=R_bw_upd_prev';
        feet=zeros(12,1);
        for i=1:4
            if(contact_legs(i)>=1)
                feet(3*(i-1)+1:3*i)=pred_state(10+3*(i-1)+1:10+3*i)+delta_x(9+3*(i-1)+1:9+3*i);
                if (assumeFlatFloor)
                    feet(3*i)=0.0327;
                end
            else
                feet(3*(i-1)+1:3*i)=pred_state(10+3*(i-1)+1:10+3*i);
            end
        end
        p_fl_upd_prev_w(:,t+1)=feet(1:3);
        p_fr_upd_prev_w(:,t+1)=feet(4:6);
        p_rl_upd_prev_w(:,t+1)=feet(7:9);
        p_rr_upd_prev_w(:,t+1)=feet(10:12);
        bias_acc=pred_state(23:25)+delta_x(22:24);
        bias_gyro=pred_state(26:28)+delta_x(25:27);
        pcov_upd_prev=upd_state_cov;
    end
end
if(only_predict)
    plot_str='dead-reckoning';
else
    plot_str='filtered';
end
%% position plots
h(1)=figure;
plot(time_gt,r_gt_w(1,:),'r-','LineWidth',2);
hold on;
plot(time_gt,r_upd_prev_w(1,:),'b-','LineWidth',2);
title('position COM x');
legend('ground truth',plot_str);
xlabel('time in seconds');
ylabel('distance in meters');

h(2)=figure;
plot(time_gt,r_gt_w(2,:),'r-','LineWidth',2);
hold on;
plot(time_gt,r_upd_prev_w(2,:),'b-','LineWidth',2);
title('position COM y');
legend('ground truth',plot_str);
xlabel('time in seconds');
ylabel('distance in meters');

h(3)=figure;
plot(time_gt,r_gt_w(3,:),'r-','LineWidth',2);
hold on;
plot(time_gt,r_upd_prev_w(3,:),'b-','LineWidth',2);
title('position COM z');
legend('ground truth',plot_str);
xlabel('time in seconds');
ylabel('distance in meters');

%% velocity plots
h(4)=figure;
plot(time_gt,v_gt_w(1,:),'r-','LineWidth',2);
hold on;
plot(time_gt,v_upd_prev_w(1,:),'b-','LineWidth',2);
title('velocity COM x');
legend('ground truth',plot_str);
xlabel('time in seconds');
ylabel('distance in meters');

h(5)=figure;
plot(time_gt,v_gt_w(2,:),'r-','LineWidth',2);
hold on;
plot(time_gt,v_upd_prev_w(2,:),'b-','LineWidth',2);
title('velocity COM y');
legend('ground truth',plot_str);
xlabel('time in seconds');
ylabel('distance in meters');

h(6)=figure;
plot(time_gt,v_gt_w(3,:),'r-','LineWidth',2);
hold on;
plot(time_gt,v_upd_prev_w(3,:),'b-','LineWidth',2);
title('velocity COM z');
legend('ground truth',plot_str);
xlabel('time in seconds');
ylabel('distance in meters');

%% rotation plots
[roll_upd,pitch_upd,yaw_upd]=quat2rpy(q_upd_prev_b);
h(7)=figure;
plot(time_gt,roll_gt,'r-','LineWidth',2);
hold on;
plot(time_gt,roll_upd,'b-','LineWidth',2);
title('rotation COM roll');
legend('ground truth',plot_str);
xlabel('time in seconds');
ylabel('distance in meters');

h(8)=figure;
plot(time_gt,pitch_gt,'r-','LineWidth',2);
hold on;
plot(time_gt,pitch_upd,'b-','LineWidth',2);
title('rotation COM pitch');
legend('ground truth',plot_str);
xlabel('time in seconds');
ylabel('distance in meters');

h(9)=figure;
plot(time_gt,yaw_gt,'r-','LineWidth',2);
hold on;
plot(time_gt,yaw_upd,'b-','LineWidth',2);
title('rotation COM yaw');
legend('ground truth',plot_str);
xlabel('time in seconds');
ylabel('distance in meters');

%% save the figures
if(only_predict)
    savefig(h,'only_predict.fig');
else
    savefig(h,'filtered.fig');
end

    function [gt_mean_mod, obs_sim_mod]= modifyDataFreq(gt_mean, obs_sim, freq)
        def_freq=10000;
        gt_mean_mod=gt_mean(:,1:def_freq/freq:end);
        obs_sim_mod=obs_sim(:,1:def_freq/freq:end);
    end

    function F=error_dynamics_matrix(dt,f1_lin, f2_lin, w_lin,R_wb_upd_prev)
        F=eye(27,27);
        F(1:3,4:6)=dt*eye(3);
        F(1:3,7:9)=-R_wb_upd_prev*skewsym_op(((dt^2)/2)*f1_lin);
        F(1:3,22:24)=-((dt^2)/2)*R_wb_upd_prev;
        F(4:6,7:9)=-R_wb_upd_prev*skewsym_op(dt*(f2_lin));
        F(4:6,22:24)=-dt*R_wb_upd_prev;
        F(7:9,7:9)=rodrigues_formula(0,w_lin,dt)';%gamma(0,w_lin,dt)';
        F(7:9,25:27)=-rodrigues_formula(1,w_lin,dt)';
    end
    
    function G=rodrigues_formula(num,w,dt)
        angle=norm(dt*w);
        axis=(dt*w)/angle;
        switch (num)
            case 0
                if (angle>1e-5)
                    G=eye(3)+skewsym_op(axis)*(sin(angle))+(skewsym_op(axis)*skewsym_op(axis))*(1-cos(angle));
                else
                    G=eye(3);
                end
            case 1
                if (angle>1e-5)
                    G=eye(3)+(skewsym_op(axis)/angle)*(1-cos(angle))+((angle-sin(angle))/angle)*(skewsym_op(axis)*skewsym_op(axis));
                else
                    G=eye(3);
                end
                G=dt*G;
            case 2
                if (angle>1e-5)
                    G=eye(3)+ ((angle-sin(angle))/(angle^2))*skewsym_op(axis)+((cos(angle)-1)/(angle^2)+ 1/2)*(skewsym_op(axis)*skewsym_op(axis));
                else
                    G=eye(3);
                end
                G=(dt^2)*G;
            case 3
                if(angle>1e-5)
                    G=eye(3)+((cos(angle)-1)/(angle^3)+ 1/(2*angle))*skewsym_op(axis)+((sin(angle)-angle)/(angle^3)+1/6)*(skewsym_op(axis)*skewsym_op(axis));
                else
                    G=eye(3);
                end
                G=(dt^3)*G;
        end        
    end

    function skew_sym=skewsym_op(vec_3)
        skew_sym=[0 -1*vec_3(3) vec_3(2);
                  vec_3(3) 0 -1*vec_3(1);
                  -1*vec_3(2) vec_3(1) 0];
    end

    function [roll,pitch,yaw]=quat2rpy(q)
        % roll
        sinr_cosp=2*(q(1,:).*q(2,:)+q(3,:).*q(4,:));
        cosr_cosp=1-2*(q(2,:).*q(2,:)+q(3,:).*q(3,:));
        roll=atan2(sinr_cosp,cosr_cosp);
        % pitch
        sinp=2*(q(1,:).*q(3,:)-q(4,:).*q(2,:));
        if (abs(sinp)>1)
            pitch=sign(sinp)*pi/2;
        else
            pitch=asin(sinp);
        end
        % yaw
        siny_cosp=2*(q(1,:).*q(4,:)+q(2,:).*q(3,:));
        cosy_cosp=1-2*(q(3,:).*q(3,:)+q(4,:).*q(4,:));
        yaw=atan2(siny_cosp,cosy_cosp);
    end

    function G=gamma(k,w,dt)
        b=mod(k,2);
        m=(k-b)/2;
        wNorm=norm(w);
        factor1=0;
        factor2=0;
        %Get skew sym matrices
        wk=skewsym_op(w);
        wk2=wk*wk;
        
        %compute first factor
        if(wNorm*dt>=1e-5*sqrt((2*m+3)*(2*m+4)))
            factor1=cos(wNorm*dt);
            for i=1:m
                factor1=factor1+(-1^i)*((wNorm*dt)^(2*i)/factorial(2*i));
            end
            factor1=factor1*((-1^(m+1))/(wNorm^(2+2*m)));
        else
            factor1=(dt^(2*m+2))/factorial(2*m+2);
        end
        
        %compute second factor
        if(wNorm*dt>=1e-5*sqrt((2*m+2*b+2)*(2*m+2*b+3)))
            factor2=sin(wNorm*dt);
            for i=1:m+b-1
                factor2=factor2+(-1^i)*((wNorm*dt)^(2*i+1))/factorial(2*i+1);
            end
            factor2=factor2*((-1^(m+b))/(wNorm^(1+2*m+2*b)));
        else
            factor2=(dt^(1+2*m+2*b))/factorial(1+2*m+2*b);
        end
        
        if(b==0)
            G=(dt^k)/factorial(k)*eye(3)+factor1*wk2+factor2*wk;
        else
            G=(dt^k)/factorial(k)*eye(3)+factor1*wk+factor2*wk2;
        end
    end

    function Q=process_noise_cov(dt, noise_cov, R_wb,ang_vel_corrected )
        Q=zeros(27,27);
        Q(1:3,1:3)=((dt^3)/3)*noise_cov.Q_f+ ((dt^5)/5)*noise_cov.Q_bf +dt*noise_cov.pred_r;
        Q(1:3,4:6)=((dt^2)/2)*noise_cov.Q_f+ ((dt^4)/8)*noise_cov.Q_bf;
        Q(1:3,22:24)=-((dt^3)/6)*R_wb*noise_cov.Q_bf;
        Q(4:6,1:3)=((dt^2)/2)*noise_cov.Q_f+((dt^4)/8)*noise_cov.Q_bf;
        Q(4:6,4:6)=dt*noise_cov.Q_f+ ((dt^3)/3)*noise_cov.Q_bf +dt*noise_cov.pred_v;
        Q(4:6,22:24)=-((dt^2)/2)*R_wb*noise_cov.Q_bf;
        G3=rodrigues_formula(3,ang_vel_corrected,dt);
        G2=rodrigues_formula(2,ang_vel_corrected,dt);
        Q(7:9,7:9)=dt*noise_cov.Q_w+dt*noise_cov.pred_q+(G3+G3')*noise_cov.Q_bw;
        Q(7:9,25:27)=-G2'*noise_cov.Q_bf;
        Q(10:12,10:12)=dt*R_wb*noise_cov.Q_p*R_wb';
        Q(13:15,13:15)=dt*R_wb*noise_cov.Q_p*R_wb';
        Q(16:18,16:18)=dt*R_wb*noise_cov.Q_p*R_wb';
        Q(19:21,19:21)=dt*R_wb*noise_cov.Q_p*R_wb';
        Q(22:24,1:3)=-((dt^3)/6)*noise_cov.Q_bf*R_wb';
        Q(22:24,4:6)=-((dt^2)/2)*noise_cov.Q_bf*R_wb;
        Q(22:24,22:24)=dt*noise_cov.Q_bf;
        Q(25:27,7:9)=-noise_cov.Q_bw*G2;
        Q(25:27,25:27)=dt*noise_cov.Q_bw;
    end

    function end_effector = forward_kinematics(s, u, k, body)
        % from center of the shoulder, 
        l1 = body.upper_length;
        l2 = body.lower_length;
        end_effector = [ - l2*sin(k + u)       -  l1*sin(u);
                     sin(s)*(l2*cos(k + u) +  l1*cos(u));
                    -cos(s)*(l2*cos(k + u) +  l1*cos(u))   ];
    end

    function J=computeJacobian(s,u,k,body)
        delta=0.01;
        J1=(forward_kinematics(s+delta,u,k,body)-forward_kinematics(s-delta,u,k,body))./(2*delta);
        J2=(forward_kinematics(s,u+delta,k,body)-forward_kinematics(s,u-delta,k,body))./(2*delta);
        J3=(forward_kinematics(s,u,k+delta,body)-forward_kinematics(s,u,k-delta,body))./(2*delta);
        J=[J1 J2 J3];
    end

    function measured_s=measured_kin(angles,body, noise_cov)
        fl_ang_s = angles(1); fl_ang_u = angles(2); fl_ang_k = angles(3);
        fr_ang_s = angles(4); fr_ang_u = angles(5); fr_ang_k = angles(6);
        rl_ang_s = angles(7); rl_ang_u = angles(8); rl_ang_k = angles(9);
        rr_ang_s = angles(10); rr_ang_u = angles(11); rr_ang_k = angles(12);
        
        % forward kinematics
        fl_end_effector = forward_kinematics(fl_ang_s, fl_ang_u, fl_ang_k, body); 
        rl_end_effector = forward_kinematics(rl_ang_s, rl_ang_u, rl_ang_k, body);
        fr_end_effector = forward_kinematics(fr_ang_s, fr_ang_u, fr_ang_k, body);
        rr_end_effector = forward_kinematics(rr_ang_s, rr_ang_u, rr_ang_k, body); 

        % distance from center of mass to shoulder
        fl_offset = [body.shoulder_distance;body.y_length/2+body.shoulder_size/2;0];
        fr_offset = [body.shoulder_distance;-body.y_length/2-body.shoulder_size/2;0];
        rl_offset = [-body.shoulder_distance;body.y_length/2+body.shoulder_size/2;0];
        rr_offset = [-body.shoulder_distance;-body.y_length/2-body.shoulder_size/2;0];

        n_alpha=(-1+2*rand([3,1]))*sqrt(noise_cov.R_alpha(1,1));
%         
        J_fl=computeJacobian(fl_ang_s,fl_ang_u,fl_ang_k,body);
        J_fr=computeJacobian(fr_ang_s,fr_ang_u,fr_ang_k,body);
        J_rl=computeJacobian(rl_ang_s,rl_ang_u,rl_ang_k,body);
        J_rr=computeJacobian(rr_ang_s,rr_ang_u,rr_ang_k,body);
%         
        ns_fl=(-1+2*rand([3,1]))*sqrt(noise_cov.R_s(1,1));
        n_fl=-ns_fl+J_fl*n_alpha;
        ns_fr=(-1+2*rand([3,1]))*sqrt(noise_cov.R_s(1,1));
        n_fr=-ns_fr+J_fr*n_alpha;
        ns_rl=(-1+2*rand([3,1]))*sqrt(noise_cov.R_s(1,1));
        n_rl=-ns_rl+J_rl*n_alpha;
        ns_rr=(-1+2*rand([3,1]))*sqrt(noise_cov.R_s(1,1));
        n_rr=-ns_rr+J_rr*n_alpha;
        
        % vector from center of mass to end effector in body frame(s_i)= offset+fwd_kin
        fl_com_ee=(fl_offset+fl_end_effector)+n_fl;
        fr_com_ee=(fr_offset+fr_end_effector)+n_fr;
        rl_com_ee=(rl_offset+rl_end_effector)+n_rl;
        rr_com_ee=(rr_offset+rr_end_effector)+n_rr;
        
        measured_s=[fl_com_ee;fr_com_ee;rl_com_ee;rr_com_ee];
    end

    function [inv_s,inv_u,inv_k] = inverse_kinematics(p, body)
        % from center of the shoulder
        x = p(1);
        y = p(2);
        z = p(3);
        l1 = body.upper_length;
        l2 = body.lower_length;

        tmp = y/sqrt(y*y+z*z);
        if tmp > 1
            tmp = 1;
        elseif tmp < -1
            tmp = -1;
        end
        inv_s = asin(tmp);
        tmp = (x*x+y*y+z*z-l1*l1-l2*l2)/(2*l1*l2);
        if tmp > 1
            tmp = 1;
        elseif tmp < -1
            tmp = -1;
        end
        inv_k = -acos(tmp); % always assume knee is minus

        if abs(inv_k) > 1e-4
            if abs(inv_s) > 1e-4
                temp1 = y/sin(inv_s)+(l2*cos(inv_k)+l1)*x/l2/sin(inv_k);
            else
                temp1 = -z/cos(inv_s)+(l2*cos(inv_k)+l1)*x/l2/sin(inv_k);
            end
            temp2 = -l2*sin(inv_k)-(l2*cos(inv_k)+l1)*(l2*cos(inv_k)+l1)/(l2*sin(inv_k));
            tmp = temp1/temp2;
            if tmp > 1
                tmp = 1;
            elseif tmp < -1
                tmp = -1;
            end
            inv_u = asin(tmp);
        else
            tmp = x/(-l1-l2);
            if tmp > 1
                tmp = 1;
            elseif tmp < -1
                tmp = -1;
            end
            inv_u = asin(tmp);
        end
    end

    function pcov=setInitCov(num_states)
        pcov=1e-12*eye(num_states-1);
%         pcov(1:3,1:3)=1e-4.*pcov(1:3,1:3);
%         pcov(4:6,4:6)=1e-2.*pcov(4:6,4:6);
%         pcov(7:9,7:9)=1e-4.*pcov(7:9,7:9);
%         pcov(10:12,10:12)=1e-2.*pcov(10:12,10:12);
%         pcov(13:15,13:15)=1e-2.*pcov(13:15,13:15);
%         pcov(16:18,16:18)=1e-2.*pcov(16:18,16:18);
%         pcov(19:21,19:21)=1e-2.*pcov(19:21,19:21);
%         pcov(22:24,22:24)=1e-4*pcov(22:24,22:24);
%         pcov(25:27,25:27)=1e-4*pcov(25:27,25:27);
    end

    function noise_cov=setNoiseCov()
        noise_cov.Q_f=diag([1e-8,1e-8,1e-8]);
        noise_cov.Q_w=diag([1e-12,1e-12,1e-12]);
        noise_cov.Q_bf=diag([1e-12,1e-12,1e-12]);%should be less
        noise_cov.Q_bw=diag([1e-12,1e-12,1e-8]);
        noise_cov.Q_p=diag([1e-8,1e-8,1e-8]);

        noise_cov.pred_r=diag([1e-8,1e-8,1e-8]);
        noise_cov.pred_v=diag([1e-4,1e-4,1e-4]);
        noise_cov.pred_q=diag([1e-8,1e-8,1e-8]);

        noise_cov.R_s=diag([1e-2,1e-2,1e-2]);
        noise_cov.R_alpha=diag([1e-2,1e-2,1e-2]);
    end

if(only_predict)
   r_upd_prev_w(:,end)-r_gt_w(:,end)
else
    measurement_residual
    r_upd_prev_w(:,end)-r_gt_w(:,end)
end
end