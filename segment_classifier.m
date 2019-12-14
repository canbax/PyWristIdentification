clear all; close all;

pathData = 'C:\Users\matk0001\Documents\MATLAB\dataSet\superPixelSampleData\';
fprintf('loading data #1...\n');
tic
P1 = load(strcat(pathData,'positive1p.mat')); % features skin 1
toc
tic
N1 = load(strcat(pathData,'negative1p.mat')); % features non-skin 1
toc

fprintf('loading data#2...\n');
tic
P2 = load(strcat(pathData,'positive2p.mat')); % features skin 2
toc
tic
N2 = load(strcat(pathData,'negative2p.mat')); % features non-skin 2
toc

fprintf('#1 positive samples num %d:\n',size(P1.dataPos,1));
fprintf('#1 negative samples num %d:\n',size(N1.dataNeg,1));
fprintf('#2 positive samples num %d:\n',size(P2.dataPos,1));
fprintf('#2 negative samples num %d:\n',size(N2.dataNeg,1));

data1 = [P1.dataPos; N1.dataNeg];
data2 = [P2.dataPos; N2.dataNeg];

for i=1:size(data1,1)
    if(data1(i,end) == -9999)
        target1{i} = 'skin';
    else
        target1{i} = 'non';
    end
end
for i=1:size(data2,1)
    if(data2(i,end) == -9999)
        target2{i} = 'skin';
    else
        target2{i} = 'non';
    end
end

cost.ClassNames = {'skin', 'non'}
cost.ClassificationCosts = [0 5; 1 0]

target1 = target1';
target2 = target2';

fprintf('training ensemble #1\n');
tic
DTree = templateTree('MinLeaf',1,'MinParent',4);
Ensemble1=fitensemble(data1,target1,'Bag',30,DTree,'type','classification','Cost',cost)
toc

figure; hold on;
plot(loss(Ensemble1,data1,target1,'mode','cumulative'));
plot(oobLoss(Ensemble1,'mode','cumulative'),'k--');
xlabel('Number of trees');
ylabel('Classification error');

fprintf('Ensembel #1 predicting values on Set #2\n')
tic
[label,score] = predict(Ensemble1,data2);
[C1,order] = confusionmat(label,target2);
order
C1
toc

fprintf('training ensemble #2\n');
tic
DTree = templateTree('MinLeaf',1,'MinParent',4);
Ensemble2=fitensemble(data2,target2,'Bag',30,DTree,'type','classification','Cost',cost)
toc

figure; hold on;
plot(loss(Ensemble2,data2,target2,'mode','cumulative'));
plot(oobLoss(Ensemble2,'mode','cumulative'),'k--');
xlabel('Number of trees');
ylabel('Classification error');

fprintf('Ensembel #2 predicting values on Set #1\n')
tic
[label,score] = predict(Ensemble2,data1);
[C2,order] = confusionmat(label,target1);
order
C2
toc