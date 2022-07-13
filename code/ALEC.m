clear all;
close all;

% load('C:\dataset\mat\Iris.mat');
% X = iris(:, 1:4);
% Y = iris(:, 5);

x = load('reduction10.csv');
y = load('y_label.csv');
% Reframing labels
for mal = 1:20000
    y(mal)= 0;
end
RandIndex = randperm(length(y));
X = x(RandIndex,:);
Y = y(RandIndex);


% Parameter setting
dcRate = 0.1;% r
block = 3;
teach1block = 3;% block size threshold < 3
teach = 20;

[n, d] = size(X);
Dists = manhattanDist(X, X); % Distance measure
maxDist = max(max(Dists));
dc = dcRate * maxDist;
% percent=0.03;
% teach=n*percent;

rho = zeros(n, 1);

% Two methods of calculating density: Gaussian kernel and Cut-off kernel 
% Gaussian kernel
% for i = 1:n-1
%     for j = i+1:n
%         rho(i) = rho(i) + exp(-(Dists(i, j)/dc)*(Dists(i, j)/dc));
%         rho(j) = rho(j) + exp(-(Dists(i, j)/dc)*(Dists(i, j)/dc));
%     end
% end

% Compute rho, where rho >= 0, namely, Cut-off kernal
for i = 1:n
    rho(i) = sum(Dists(i, :) < dc) - 1; 
end

% Cut-off kernal
% for i=1:n-1
%     for j=i+1:n
%         if Dists(i, j) < dc
%             rho(i) = rho(i) + 1.;
%             rho(j) = rho(j) + 1.;
%         end
%     end
% end

delta = zeros(n, 1);
master = -ones(n, 1);
[~, ordrho] = sort(rho, 'descend');
delta(ordrho(1)) = maxDist;
for i = 2:n
    delta(ordrho(i)) = maxDist;
    for j = 1:i-1
        if Dists(ordrho(i), ordrho(j)) < delta(ordrho(i))
            delta(ordrho(i)) = Dists(ordrho(i), ordrho(j));
            master(ordrho(i)) = ordrho(j);
        end
    end
end
gamma = rho .* delta;
[~, desInd] = sort(gamma, 'descend');
isClassified = false(n, 1);
tblock = block;
numTeach = 0;
numPredict = 0;
numVote = 0;
while true
    % Compute centers  
    centers = desInd(1:tblock, 1);
    disp('centers');
    disp(centers');
    % cluster with centers
    cl = -ones(n, 1);
    cl(centers) = 1:tblock;
    
    for i = 1:n
        if cl(ordrho(i)) == -1
            cl(ordrho(i)) = cl(master(ordrho(i)));
        end
    end
    clusterIndices = centers(cl);
    
    % Compute Block information
    blockInfo = cell(tblock, 1);
    
    for i = 1:tblock
%         tEle = sum(clusterIndices == centers(i));
        blockInfo{i} = find(clusterIndices == centers(i));
    end
    
    tBlockProcessed = false(tblock, 1);
    tUnProcessedBlocks = 0;
    for i = 1:tblock
        if ismember(0, isClassified(blockInfo{i}))
            tUnProcessedBlocks = tUnProcessedBlocks + 1;
        else
            tBlockProcessed(i) = 1;
        end
    end
    
    % Step 2.3.1
    for i = 1:tblock
        if tBlockProcessed(i)
            continue;
        end
        tn = length(blockInfo{i});
        if tn < teach1block
            for j = 1:tn
                if ~isClassified(blockInfo{i}(j))
                    if numTeach >= teach
                        break;
                    end
                    predictedLabels(blockInfo{i}(j)) = Y(blockInfo{i}(j));
                    isClassified(blockInfo{i}(j)) = true;
               
                    numTeach = numTeach + 1;
                end
            end
        end
        ordPrio = desInd(clusterIndices(desInd) == centers(i));
        disp(['Ord Prio Length: ', int2str(length(ordPrio))]);
        iind = find(~isClassified(ordPrio));
        notClassifiedLen = size(iind);
        tNumTeach = 0;
        
        for j = 1:tn
            if isClassified(ordPrio(j))
                continue;
            end
            if numTeach >= teach
                break;
            end
            % Buy labels
            predictedLabels(ordPrio(j)) = Y(ordPrio(j));
            isClassified(ordPrio(j)) = true;
            numTeach = numTeach + 1;
            tNumTeach = tNumTeach + 1;
            if tNumTeach >= teach1block
                break;
            end
        end
    end
    
    for i = 1:tblock
        if tBlockProcessed(i)
            continue;
        end
        
%         for j = 1:tn
%             if isClassified(blockInfo{i}(j))
%                 if tFirstLabel
%                     tLabel = predictedLabels(blockInfo{i}(j));
%                     tFirstLabel = false;
%                 elseif tLabel ~= predictedLabels(blockInfo{i}(j))
%                     tPure = false;
%                     break;
%                 end
%             end
%         end
        % Is pure
        iind = find(isClassified(blockInfo{i}));
        isUniqueLabel = length(unique(predictedLabels(blockInfo{i}(iind))));
        
        if isUniqueLabel == 1
            notiind = find(~isClassified(blockInfo{i}));
            predictedLabels(blockInfo{i}(notiind)) = predictedLabels(blockInfo{i}(iind(1)));
            isClassified(blockInfo{i}(notiind)) = true;
            numPredict = numPredict + length(notiind);
        end
    end
    tblock = tblock + 1;
    if tUnProcessedBlocks == 0
        break;
    end
    if numTeach >= teach
        break;
    end
end

%% Vote
% Reclustering
tblock = max(predictedLabels);
% Compute centers
[~, centers] = sort(gamma, 'descend');
centers = centers(1:tblock);
% Cluster with centers
cl = -ones(n, 1);
cl(centers) = 1:tblock;

for i = 1:n
    if cl(ordrho(i)) == -1
        cl(ordrho(i)) = cl(master(ordrho(i)));
    end
end
clusterIndices = centers(cl);

% Compute Block information
blockInfo = cell(tblock, 1);
for i = 1:tblock
    tEle = sum(clusterIndices == centers(i));
    blockInfo{i} = find(clusterIndices == centers(i));
end

for i = 1:tblock
    labels = unique(predictedLabels(blockInfo{i}));
    tLen = length(labels);
    vote = zeros(length(labels), 1);
    count = 1;
    for j = 1:tLen
        vote(count, 1) = sum(predictedLabels(blockInfo{i}) == labels(j));
        count = count + 1;
    end
    [~, ti] = max(vote);
    iind = find(predictedLabels(blockInfo{i}) == -1);
    predictedLabels(blockInfo{i}(iind)) = labels(ti);
end

errors=0;
for i=1:n
   if predictedLabels(i)~=Y(i)
      errors=errors+1;
   end
end
accracy=(n-errors-numTeach)/(n-numTeach);
disp('accracy:');
disp(accracy);



