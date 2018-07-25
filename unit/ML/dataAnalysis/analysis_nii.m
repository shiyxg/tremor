load('/home/shi/Data.mat')

marks = double(marks);
record = double(record);
marks = permute(marks,[1,3,2]);
record = permute(record,[1,3,2]);
%data = int32((record+3)/3*256);
data = marks*5+record;
[XT,YT,TT] = size(data);
hist(reshape(data(:,:,:),[XT*YT*TT,1]),2)
for i = 1:YT
    %hist(reshape(data(i,:,:),[YT*TT,1]),2)
    pcolor(reshape(data(i,:,:),[YT,TT])')
    title(i)
    shading interp
    set(gca,'ydir','reverse')
    pause(0.1)
end
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
