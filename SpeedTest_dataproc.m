load('speed_test13.mat');

dt = 1/60;
time_stamp = time_stamp - time_stamp(1);
circle_sum = double(circle_sum); %(1:512));
%dc = sum(circle_sum)/length(circle_sum);
%circle_sum = circle_sum - dc;
mode_flag = 1 - double(mode_flag);
power_sum = double(power_sum);
P_ratio_1d = circle_sum./power_sum;

figure(111); hold on;
plot(power_sum./(max(power_sum)), 'r');
plot(circle_sum./(max(circle_sum)), 'b');
plot(P_ratio_1d/(max(P_ratio_1d)), 'g');

img_list = 1.0*double(img_list);

figure(2); %plot(time, (power_sum)/(1.0*max(power_sum)), '-bo'); 
hold on; 
plot(abs(fft((circle_sum)/(1.0*max(circle_sum)))), '-go'); 
hold on; 
figure(3);
plot(cx);
hold on;
plot(cy);


size_tmp_1d = size(img_list);
img_count = size_tmp_1d(3);
for i_index = 115:img_count

    figure(5); hold off;
    imagesc(squeeze(img_list(:,:,i_index))); %shading interp
    axis tight equal
    hold on;
    plot(cy(i_index) + 1, cx(i_index) + 1, 'r+');
end



%plot(time+4*dt, ((mod_flag)/2.0)+0.25, ':r*'); ylim([-0.1 1.1]);