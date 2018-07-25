%load('/home/shi/FaultDetection/data.mat');
%data = databk;
%% plot 
% 
% Filename = ['xt.gif'];
% 
% 
% for i = 1:301
%     hist(reshape(data(i,:,:),[251*2501,1]),1000)
%     title(i);
%     xlabel('data')
%     ylabel('times(1/1000)')
%     xlim([-1.5e4,1.5e4])
%     ylim([0,4e4]);
%     %pause(0.1)
%     
%     f=getframe(gcf);  
%     imind=frame2im(f);
%     [imind,cm] = rgb2ind(imind,256);
%     if i==1
%         imwrite(imind,cm,Filename,'gif', 'Loopcount',inf,'DelayTime',0.05);%第一次必须创建！
%     else
%         imwrite(imind,cm,Filename,'gif','WriteMode','append','DelayTime',0.05);
%     end
% end

%% data -> image(256)
databk = data;
for i = 1:301
    for j = 1:251
        for k = 1:2501
            if (data(i,j,k)< 0.5e4) && (data(i,j,k) > -0.5e4)
                data(i,j,k) = floor((data(i,j,k)+0.5e4)/1e4 *256);
            elseif data(i,j,k)>=0.5e4
                data(i,j,k) = 255;
            elseif data(i,j,k)<=-0.5e4
                data(i,j,k) = 0;
            end
        end
    end
end
a = reshape(data(:,1,:),[301,2501]);
b = reshape(databk(:,1,:),[301,2501]);
figure(1)
subplot(1,2,1)
pcolor(a');
shading interp
set(gca,'yDir','reverse')
subplot(1,2,2)
pcolor(b');
shading interp
set(gca,'yDir','reverse')


%% data -> pixel(32*32 channels)

data = data(:,1,:);
[YL,XL,TL] = size(data);
cache = zeros(size(data)+31);
cache(16:(16+YL-1),16:(16+XL-1),16:(16+TL-1)) = data;
data = zeros(YL,XL,TL,32,32);
for i = 1:YL
    for j = 1:XL
        for k = 1:TL
            data(i,j,k,:,:) = cache(i:(i+31),j:(j+31),k);
            i,j,k
        end
    end
end






