%%
%MCP
%éUïzê}ÅCÉqÉXÉgÉOÉâÉÄ

E1 = load_raw('C:\Users\yourb\Desktop\NZdata\MCP\3Dvolume\PBL_MCP_phantom12E1.raw','*single');
E2 = load_raw('C:\Users\yourb\Desktop\NZdata\MCP\3Dvolume\PBL_MCP_phantom12E2.raw','*single');
E3 = load_raw('C:\Users\yourb\Desktop\NZdata\MCP\3Dvolume\PBL_MCP_phantom12E3.raw','*single');
E4 = load_raw('C:\Users\yourb\Desktop\NZdata\MCP\3Dvolume\PBL_MCP_phantom12E4.raw','*single');
GT = load_raw('C:\Users\yourb\Desktop\NZdata\MCP\Groundtruth0.50.25robust.raw','*single');

%{
E1 = load_raw('C:\Users\yourb\Desktop\NZdata\MCP\3Dvolume\MCP_phantom12E1.raw','*single');
E2 = load_raw('C:\Users\yourb\Desktop\NZdata\MCP\3Dvolume\MCP_phantom12E2.raw','*single');
E3 = load_raw('C:\Users\yourb\Desktop\NZdata\MCP\3Dvolume\MCP_phantom12E3.raw','*single');
E4 = load_raw('C:\Users\yourb\Desktop\NZdata\MCP\3Dvolume\MCP_phantom12E4.raw','*single');
GT = load_raw('C:\Users\yourb\Desktop\NZdata\MCP\Groundtruth0.50.25robust.raw','*single');
%}

siz = [544,544,50];
E1 = reshape(E1,siz); E2 = reshape(E2,siz); E3 = reshape(E3,siz); E4 = reshape(E4,siz); GT = reshape(GT,siz);

%%
%#1Ç©ÇÁ#20Ç‹Ç≈
volsize = 5;
mask = false(siz); mask(:,:,16:20) = true;
E1train = E1(mask); E1train = reshape(E1train,[siz(1),siz(2),volsize]);
E2train = E2(mask); E2train = reshape(E2train,[siz(1),siz(2),volsize]);
E3train = E3(mask); E3train = reshape(E3train,[siz(1),siz(2),volsize]);
E4train = E4(mask); E4train = reshape(E4train,[siz(1),siz(2),volsize]);
GTtrain = GT(mask); GTtrain = reshape(GTtrain,[siz(1),siz(2),volsize]);
%%
E1train = E1train(:); E2train = E2train(:); E3train = E3train(:); E4train = E4train(:); 
GTtrain = GTtrain(:);

%%
mat1E1 = E1train(GTtrain == 1);
mat1E2 = E2train(GTtrain == 1);
mat1E3 = E3train(GTtrain == 1);
mat1E4 = E4train(GTtrain == 1);

mat2E1 = E1train(GTtrain == 2);
mat2E2 = E2train(GTtrain == 2);
mat2E3 = E3train(GTtrain == 2);
mat2E4 = E4train(GTtrain == 2);

mat3E1 = E1train(GTtrain == 3);
mat3E2 = E2train(GTtrain == 3);
mat3E3 = E3train(GTtrain == 3);
mat3E4 = E4train(GTtrain == 3);

mat4E1 = E1train(GTtrain == 4);
mat4E2 = E2train(GTtrain == 4);
mat4E3 = E3train(GTtrain == 4);
mat4E4 = E4train(GTtrain == 4);

mat5E1 = E1train(GTtrain == 5);
mat5E2 = E2train(GTtrain == 5);
mat5E3 = E3train(GTtrain == 5);
mat5E4 = E4train(GTtrain == 5);

mat6E1 = E1train(GTtrain == 6);
mat6E2 = E2train(GTtrain == 6);
mat6E3 = E3train(GTtrain == 6);
mat6E4 = E4train(GTtrain == 6);

mat7E1 = E1train(GTtrain == 7);
mat7E2 = E2train(GTtrain == 7);
mat7E3 = E3train(GTtrain == 7);
mat7E4 = E4train(GTtrain == 7);

mat8E1 = E1train(GTtrain == 8);
mat8E2 = E2train(GTtrain == 8);
mat8E3 = E3train(GTtrain == 8);
mat8E4 = E4train(GTtrain == 8);

mat9E1 = E1train(GTtrain == 9);
mat9E2 = E2train(GTtrain == 9);
mat9E3 = E3train(GTtrain == 9);
mat9E4 = E4train(GTtrain == 9);

mat10E1 = E1train(GTtrain == 10);
mat10E2 = E2train(GTtrain == 10);
mat10E3 = E3train(GTtrain == 10);
mat10E4 = E4train(GTtrain == 10);

mat11E1 = E1train(GTtrain == 11);
mat11E2 = E2train(GTtrain == 11);
mat11E3 = E3train(GTtrain == 11);
mat11E4 = E4train(GTtrain == 11);

mat12E1 = E1train(GTtrain == 12);
mat12E2 = E2train(GTtrain == 12);
mat12E3 = E3train(GTtrain == 12);
mat12E4 = E4train(GTtrain == 12);
%%
pts = 0.1:0.005:0.7;
hold on;
plot(pts,ksdensity(mat1E1,pts,'Function','pdf'));
plot(pts,ksdensity(mat2E1,pts,'Function','pdf'));
plot(pts,ksdensity(mat3E1,pts,'Function','pdf'));
plot(pts,ksdensity(mat4E1,pts,'Function','pdf'));
plot(pts,ksdensity(mat5E1,pts,'Function','pdf'));
plot(pts,ksdensity(mat6E1,pts,'Function','pdf'));
plot(pts,ksdensity(mat7E1,pts,'Function','pdf'));
plot(pts,ksdensity(mat8E1,pts,'Function','pdf'));
plot(pts,ksdensity(mat9E1,pts,'Function','pdf'));
plot(pts,ksdensity(mat10E1,pts,'Function','pdf'));
plot(pts,ksdensity(mat11E1,pts,'Function','pdf'));
plot(pts,ksdensity(mat12E1,pts,'Function','pdf'));
hold off;
%%
%I
scatter(mat7E1,mat7E2,'.'); hold on
scatter(mat12E1,mat12E2,'.'); hold on
scatter(mat10E1,mat10E2,'.');

scatter(mat7E1,mat7E3,'.'); hold on
scatter(mat12E1,mat12E3,'.'); hold on
scatter(mat10E1,mat10E3,'.');

scatter(mat7E1,mat7E4,'.'); hold on
scatter(mat12E1,mat12E4,'.'); hold on
scatter(mat10E1,mat10E4,'.'); hold off
axis equal tight
xlabel('Energy1')
ylabel('Energy2,Energy3,Energy4')
xlim([0.1,0.5])
ylim([0.1,0.5])
legend('I4.5,E1&E2','I9,E1&E2','I18,E1&E2',...
    'I4.5,E1&E3','I9,E1&E3','I18,E1&E3',...
    'I4.5,E1&E4','I9,E1&E4','I18,E1&E4','Location','northwest')

%%
%Au
scatter(mat9E1,mat9E2,'.'); hold on
scatter(mat8E1,mat8E2,'.'); hold on
scatter(mat11E1,mat11E2,'.');

scatter(mat9E1,mat9E3,'.'); hold on
scatter(mat8E1,mat8E3,'.'); hold on
scatter(mat11E1,mat11E3,'.');
 
scatter(mat9E1,mat9E4,'.'); hold on
scatter(mat8E1,mat8E4,'.'); hold on
scatter(mat11E1,mat11E4,'.'); hold off
axis equal tight
xlabel('Energy1')
ylabel('Energy2,Energy3,Energy4')
xlim([0.1,0.4])
ylim([0.1,0.4])
legend('Au2,E1&E2','Au4,E1&E2','Au8,E1&E2',...
    'Au2,E1&E3','Au4,E1&E3','Au8,E1&E3',...
    'Au2,E1&E4','Au4,E1&E4','Au8,E1&E4','Location','northwest')

%%
%Cd
scatter(mat4E1,mat4E2,'.'); hold on
scatter(mat6E1,mat6E2,'.');

scatter(mat4E1,mat4E3,'.'); hold on
scatter(mat6E1,mat6E3,'.');

scatter(mat4E1,mat4E4,'.'); hold on
scatter(mat6E1,mat6E4,'.'); hold off
axis equal tight
xlabel('Energy1')
ylabel('Energy2,Energy3,Energy4')
xlim([0.1,0.35])
ylim([0.1,0.35])
legend('Cd2,E1&E2','Cd8,E1&E2',...
    'Cd2,E1&E3','Cd8,E1&E3',...
    'Cd2,E1&E4','Cd8,E1&E4','Location','northwest')

%%
%Water Lipid
scatter(mat3E1,mat3E2,'.'); hold on
scatter(mat5E1,mat5E2,'.');

scatter(mat3E1,mat3E3,'.'); hold on
scatter(mat5E1,mat5E3,'.');

scatter(mat3E1,mat3E4,'.'); hold on
scatter(mat5E1,mat5E4,'.'); hold off
axis equal tight
xlabel('Energy1')
ylabel('Energy2,Energy3,Energy4')
xlim([0.1,0.25])
ylim([0.1,0.25])
legend('Water,E1&E2','Lipid,E1&E2',...
    'Water,E1&E3','Lipid,E1&E3',...
    'Water,E1&E4','Lipid,E1&E4','Location','northwest')

%%
%HA
scatter(mat2E1,mat2E2,'.'); hold on
scatter(mat1E1,mat1E2,'.');

scatter(mat2E1,mat2E3,'.'); hold on
scatter(mat1E1,mat1E3,'.');

scatter(mat2E1,mat2E4,'.'); hold on
scatter(mat1E1,mat1E4,'.'); hold off
axis equal tight
xlabel('Energy1')
ylabel('Energy2,Energy3,Energy4')
xlim([0.1,0.7])
ylim([0.1,0.7])
legend('HA200,E1&E2','HA800,E1&E2',...
    'HA200,E1&E3','HA800,E1&E3',...
    'HA200,E1&E4','HA800,E1&E4','Location','northwest')

%%
out0 = E4train(GTtrain == 4);
out1 = E4train(GTtrain == 7);
out2 = E4train(GTtrain == 9);
out3 = E4train(GTtrain == 8);
out4 = E4train(GTtrain == 11);
out5 = E4train(GTtrain == 12);
edges = [0.1 0.1:0.01:1.2 1.2];
%xlim([-1.0 0.3]);

histogram(out0,edges,'Normalization','probability');
hold on
histogram(out1,edges,'Normalization','probability');
hold on
histogram(out2,edges,'Normalization','probability');
hold on
histogram(out3,edges,'Normalization','probability');
hold on
histogram(out4,edges,'Normalization','probability');
hold on
histogram(out5,edges,'Normalization','probability');
%hold off
%legend('Water(No.4)','I9(No.7)','I18(No.9)','HA100(No.8)','HA400(No.11)','HA800(No.12)');

%%
edges = [0.1 0.1:0.003:0.6 0.6];
%xlim([-1.0 0.3]);

histogram(mat1E1,edges,'Normalization','probability'); hold on
histogram(mat2E1,edges,'Normalization','probability'); hold on
histogram(mat3E1,edges,'Normalization','probability'); hold on
histogram(mat4E1,edges,'Normalization','probability'); hold on
histogram(mat5E1,edges,'Normalization','probability'); hold on
histogram(mat6E1,edges,'Normalization','probability'); hold on
histogram(mat7E1,edges,'Normalization','probability'); hold on
histogram(mat8E1,edges,'Normalization','probability'); hold on
histogram(mat9E1,edges,'Normalization','probability'); hold on
histogram(mat10E1,edges,'Normalization','probability'); hold on
histogram(mat11E1,edges,'Normalization','probability'); hold on
histogram(mat12E1,edges,'Normalization','probability'); hold on
histogram(mat13E1,edges,'Normalization','probability'); hold off
legend('Lipid(No.6)','FN400(No.3)','FN200(No.1)','FN100(No.2)','FN50(No.5)','FN25(No.25)');

