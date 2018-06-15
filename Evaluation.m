%%
y =340;
rangex = [220,290];
rangey = [y-50,y+50];
test1 = E1(:,y,34);
test2 = E1(:,y,34);


subplot(3,4,[1,6])
imagesc(E1(:,:,30)');
%imagesc(Output(:,:,2)');
%colormap(map);
colormap gray
axis equal tight
rectangle('Position',[1,y-0.5,436,1],'FaceColor','none','EdgeColor','r',...
    'LineWidth',1)
xlim(rangex);
ylim(rangey);


subplot(3,4,[3,8])
imagesc(E1(:,:,30)');
%colormap gray
axis equal tight
 rectangle('Position',[1,y-0.5,436,1],'FaceColor','none','EdgeColor','r',...
    'LineWidth',1)
xlim(rangex);
ylim(rangey);


M1 = movmean(test1,3);
subplot(3,4,[9,10])
h= plot([test1]);
h(1).Color = 'g';
xlim(rangex);
ylim([-0.1,0.7]);
legend({'Gray value'});


M2 = movmean(test2,3);
subplot(3,4,[11,12])
h= plot([test2]);
h(1).Color = 'g';
xlim(rangex);
ylim([-0.1,0.7]);
legend({'Gray value'});


%%
%Convert to HU
acE1  = mean(E1(GT== 3));
acE2  = mean(E2(GT== 3));
acE3  = mean(E3(GT== 3));
acE4  = mean(E4(GT== 3));

E1 = (E1 - acE1) / acE1*1000;
E2 = (E2 - acE2) / acE2*1000;
E3 = (E3 - acE3) / acE3*1000;
E4 = (E4 - acE4) / acE4*1000;


%%
%CT Phantom Evaluation
inin = E1;
mat = 3;
slice =26;

%mat3 = water 
tmp1 = GT== 3;
in  = inin(tmp1);
avg = mean(in);

tmp1 = GT(:,:,slice)== mat;
Dtest = bwdist(~tmp1);
in = inin(:,:,slice);
%in = (in-avg)/avg*1000;
sta = 1; en = 32.5; inter = 0.5; 
num = sta:inter:en;
avgvec = zeros(size(num,1),1);
num=1;
for dist = sta:inter:en
    tmp2 = Dtest<=dist;
    tmp3 = Dtest>(dist-inter);
    out = and(tmp1,tmp2);
    out = and(out,tmp3);
    avgvec(num,1) = mean(in(out));
    num = num+1;
end
num = sta:inter:en;
scatter(num,avgvec);
ylim([-150 100])

%%
%MD Evaluation
in = Gd;
mat = 6;
slice =30;

%mat3 = water 
tmp1 = GT(:,:,slice)== mat;
Dtest = bwdist(~tmp1);
in = in(:,:,slice);
%in = (in-avg)/avg*1000;
sta = 1; en = 32.5; inter = 0.5; 
num = sta:inter:en;
avgvec = zeros(size(num,1),1);
num=1;
for dist = sta:inter:en
    tmp2 = Dtest<=dist;
    tmp3 = Dtest>(dist-inter);
    out = and(tmp1,tmp2);
    out = and(out,tmp3);
    avgvec(num,1) = mean(in(out));
    num = num+1;
end
num = sta:inter:en;
scatter(num,avgvec');