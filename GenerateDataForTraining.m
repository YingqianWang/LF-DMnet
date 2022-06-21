%% Initialization
clear all;
clc;
%close all;
addpath(genpath('./Functions/'))


%% Parameters setting

angRes = 5;
patchsize = 152;
stride = 32;

sourceDataPath = '../Datasets/';
sourceDatasets = dir(sourceDataPath);
sourceDatasets(1:2) = [];
datasetsNum = length(sourceDatasets);
idx = 0;
SavePath = '../Data/Train_MDSR/';
if exist(SavePath, 'dir')==0
    mkdir(SavePath);
end

Train_Dataset_List = {'HCI_new', 'HCI_old', 'Stanford_Gantry'};

for DatasetIndex = 1 : length(Train_Dataset_List)
    Train_Dataset = Train_Dataset_List{DatasetIndex};
    sourceDataFolder = [sourceDataPath, Train_Dataset, '/training/'];
    folders = dir(sourceDataFolder); % list the scenes
    if isempty(folders)
        continue
    end
    folders(1:2) = [];
    sceneNum = length(folders);
    
    for iScene = 1 : sceneNum
        idx_s = 0;
        sceneName = folders(iScene).name;
        sceneName(end-3:end) = [];
        fprintf('Generating training data of Scene_%s in Dataset %s......\t', sceneName, Train_Dataset);
        dataPath = [sourceDataFolder, folders(iScene).name];
        data = load(dataPath);
        
        LF = data.LF;
        
        a_0 = (9 - angRes) / 2 + 1;
        a_t = (9 - angRes) / 2 + angRes;
        LF = LF(a_0:a_t, a_0:a_t, :, :, 1:3);
        [U, V, H, W, ~] = size(LF);
        
        for h =  1 : stride : H -  patchsize + 1
            for w =  1 : stride : W -  patchsize + 1

                subLF = single(zeros(U, V, patchsize, patchsize, 3));
                for u = 1 : U
                    for v = 1 : V
                        k = (u-1)*V + v;
                        subLF(u, v, :, :, :) = squeeze(LF(u, v, h:h+patchsize-1, w:w+patchsize-1, :));                   
                    end
                end               
                
                idx = idx + 1;
                
                SavePath_H5 = [SavePath, num2str(idx,'%06d'),'.h5'];
                h5create(SavePath_H5, '/lf', size(subLF), 'Datatype', 'single');
                h5write(SavePath_H5, '/lf', single(subLF), [1,1,1,1,1], size(subLF));
            end
        end
        fprintf([num2str(idx), ' training samples have been generated \n']);
    end
end


