%load CT volume data
fileID = fopen('InputPath_MCP.txt');
C = textscan(fileID,'%s');
fclose(fileID);
InputPath = C{1,1}; 
InputPath = cell2mat(InputPath);

fileID = fopen('CasePath_MCP.txt');
C = textscan(fileID,'%s');
fclose(fileID);
CasePath = C{1,1};

%load_data
E1 = load_raw([InputPath CasePath{1,:} '.raw'],'*single');
E2 = load_raw([InputPath CasePath{2,:} '.raw'],'*single');
E3 = load_raw([InputPath CasePath{3,:} '.raw'],'*single');
E4 = load_raw([InputPath CasePath{4,:} '.raw'],'*single');
GT = load_raw([InputPath CasePath{5,:} '.raw'],'*single');

siz = [544,544,50];  
E1 = reshape(E1,siz); 
E2 = reshape(E2,siz); 
E3 = reshape(E3,siz); 
E4 = reshape(E4,siz); 
GT = reshape(GT,siz);

%%
%load MD data
fileID = fopen('InputPath_MD.txt');
C = textscan(fileID,'%s');
fclose(fileID);
InputPath = C{1,1}; 
InputPath = cell2mat(InputPath);

fileID = fopen('CasePath_MD.txt');
C = textscan(fileID,'%s');
fclose(fileID);
CasePath = C{1,1};

%load_data
Gd = load_raw([InputPath CasePath{1,:} '.raw'],'*single');
Gold = load_raw([InputPath CasePath{2,:} '.raw'],'*single');
HA = load_raw([InputPath CasePath{3,:} '.raw'],'*single');
Iodine = load_raw([InputPath CasePath{4,:} '.raw'],'*single');
Lipid = load_raw([InputPath CasePath{5,:} '.raw'],'*single');
Water = load_raw([InputPath CasePath{5,:} '.raw'],'*single');

siz = [544,544,50];  
Gd = reshape(Gd,siz); 
Gold = reshape(Gold,siz); 
HA = reshape(HA,siz); 
Iodine = reshape(Iodine,siz); 
Lipid = reshape(Lipid,siz);
Water = reshape(Water,siz);