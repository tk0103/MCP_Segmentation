%load_path
fileID = fopen('InputPath_ano.txt');
C = textscan(fileID,'%s');
fclose(fileID);
InputPath = C{1,1}; 
InputPath = cell2mat(InputPath);

fileID = fopen('CasePath_ano.txt');
C = textscan(fileID,'%s');
fclose(fileID);
CasePath = C{1,1};

%load_data
%Img = load_raw([InputPath CasePath{1,:} '.raw'],'*single');
EdgeImg = load_raw([InputPath CasePath{2,:} '.raw'],'*single');

siz = [544 544 50];
%Img = reshape(Img,siz);
EdgeImg = reshape(EdgeImg,siz);
