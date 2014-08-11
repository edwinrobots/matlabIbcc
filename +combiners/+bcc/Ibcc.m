classdef  Ibcc < combiners.AbstractCombiner
 
    properties        
        %Experiment settings ----------------------------------------------
        settings = [] %a BccSettings object
                        
        %????
        targetWeights = []
        
        %Model Parameters -------------------------------------------------
        %Priors
        Alpha0 = []
        Nu0
        
        %Posterior HyperParams
        Alpha %final posterior alphas to be returned as agent ratings
        Nu
        
        %Posterior Params
        lnPi = []
        lnKappa = []
        
        %Data
        Tmat %targets in matrix form, columns with all-0s for unknown labels
        trainVoteCounts %counts of votes on the training points
        
        C %the last set of responses seen
        CmatByScore % set containing one binary matrix for each possible score value
        
        %Indexes of data in obj.targets etc.
        trainIdxs
        testIdxs
    end
    
    methods (Static)        
        function f = dirpdf(X, A)
            B = prod(gamma(A), 2) ./ gamma(sum(A, 2));
            f = 1./B .* prod(X.^(A-1), 2);
        end
        
        function f=logDirPdf(logX, A)
            invB = log(gamma(sum(A, 2))) - sum(log(gamma(A)), 2);
            f = invB + sum(logX.*(A-1), 2);           
        end

        function id = getId()
            id = 'IBCC';            
        end
    end
    
    methods       
        function obj = Ibcc(bccSettings, nAgents, nClasses, nScores, targets)
            obj@combiners.AbstractCombiner(nAgents, nClasses, nScores, targets);
            obj.settings = bccSettings;                              
            obj.debug = obj.settings.debug;

            obj.Nu0 = bccSettings.Nu0;
            if length(obj.Nu0)<obj.nClasses
                obj.Nu0 = [obj.Nu0 ones(1,obj.nClasses-length(obj.Nu0)).*obj.Nu0(1)];
            end
            
            obj.setAlphaPrior(bccSettings.Alpha0, nAgents);
            obj.setTargets(obj.targets); 
        end
        
        function prepareC(obj, Clist)
            C3 = round(Clist{3});
            
            if ~isempty(obj.settings.IbccMapFunction)
                mappedC3 = obj.settings.IbccMapFunction(C3);
            elseif ~isempty(obj.settings.scoreMap)
                mappedC3 = C3;
                for s=1:size(obj.settings.scoreMap,2)
                    scoreIdxs = C3==obj.settings.scoreMap(2,s);
                    mappedC3(scoreIdxs) = obj.settings.scoreMap(1, s);
                end

                [minVal, minIdx] = min(obj.settings.scoreMap(2,:));
                [maxVal, maxIdx] = max(obj.settings.scoreMap(2,:));
                mappedC3(C3<minVal) = obj.settings.scoreMap(1,minIdx);
                mappedC3(C3>maxVal) = obj.settings.scoreMap(1,maxIdx);
            else
                mappedC3 = C3 - obj.settings.minScore + 1;
            end
            Clist{3} = mappedC3;
            
            obj.CmatByScore = {};
            for l=1:obj.nScores
                obj.CmatByScore{l} = sparse(Clist{1}, Clist{2}, double(Clist{3}==l), obj.nAgents, obj.nObjects);
                obj.CmatByScore{l} = full(obj.CmatByScore{l});
            end
            
            obj.C = Clist;
        end
        
        function baseData = correctBaseData(obj, baseOutputs)
            baseData = obj.convertMatToSparse(baseOutputs);
        end    
        
        function setTargets(obj, targets)
            obj.targets = targets;
            obj.Tmat = zeros(obj.nClasses, length(targets));
            obj.trainIdxs = find(targets~=0);
            obj.testIdxs = find(targets==0);
            if isempty(obj.targetWeights)
                obj.Tmat(sub2ind([size(obj.Tmat,1) size(obj.Tmat,2)], ...
                    targets(obj.trainIdxs), obj.trainIdxs)) = 1;
            else
                 obj.Tmat(sub2ind([size(obj.Tmat,1) size(obj.Tmat,2)], ...
                    targets(obj.trainIdxs), obj.trainIdxs)) = obj.targetWeights(obj.trainIdxs);               
            end
            obj.nObjects = length(obj.targets);
        end        
                
        %use an explicit alpha prior instead of the lambda stuff
        function setAlphaPrior(obj, AlphaPrior, nDupes)
            if ~exist('nDupes','var')
                obj.Alpha0 = AlphaPrior;
            elseif size(AlphaPrior,3)==1            
                obj.Alpha0 = repmat(AlphaPrior, [1, 1, nDupes]);
            elseif size(AlphaPrior,3)<nDupes
                obj.Alpha0 = zeros(size(obj.Alpha0,1), obj.nScores, nDupes);
                obj.Alpha0(:,:,1:size(AlphaPrior,3)) = AlphaPrior;
                for a=size(AlphaPrior,3)+1:nDupes
                    obj.Alpha0(:,:,a) = sum(AlphaPrior,3) ./ size(AlphaPrior,3);
                end    
            elseif size(AlphaPrior,3)>nDupes
                obj.Alpha0 = AlphaPrior(:,:,1:nDupes);
            end
            for trusted=obj.settings.trustedAgents
                obj.Alpha0(:,:,trusted) = obj.trustedAlpha;
            end            
        end
                
        function [pT, lnpCT] = expectedT(obj)
            lnpCT = zeros(obj.nObjects, obj.nClasses);
            pT = zeros(obj.nObjects, obj.nClasses);
            
            indx = sub2ind([obj.nScores obj.nAgents], obj.C{3}, obj.C{1});                 
            lnPiIndx = obj.lnPi(:,indx);
                        
            nResp = length(obj.C{2});
            nObj = obj.nObjects;
            objIdxs = obj.C{2}';
            lnK = obj.lnKappa;
            parfor j=1:obj.nClasses
                lnpCT(:, j) = sparse(objIdxs, ones(1,nResp), lnPiIndx(j,:), nObj, 1) + lnK(j);
            end

            expA = zeros(obj.nObjects, obj.nClasses);
            rescale = 100-max(lnpCT,[],2);            
            parfor j=1:obj.nClasses    
                expAj = exp(lnpCT(:,j) + rescale);
                %stop any extreme values from causing NAN
                expA(:,j) = expAj;
            end
            expB = sum(expA,2);

            %using loops so we avoid having to repmat with large matrices
            parfor j=1:obj.nClasses
                pT(:,j) = expA(:,j)./expB;
            end
            
            if length(obj.targets)<1
                obj.post_T = pT;
            else
                obj.post_T = obj.Tmat;
                obj.post_T(:, obj.testIdxs) = pT(obj.testIdxs, :)';
            end            
        end
        
        function Count = voteCounts(obj)
            Count = zeros(obj.nClasses, obj.nScores, obj.nAgents);    
                        
            for j=1:obj.nClasses
                try
                    Tj = obj.post_T(j,:)';
                    CmatSlices = obj.CmatByScore;
                    for l=1:obj.nScores
                        Count(j, l, :) = CmatSlices{l} * Tj;
                    end
                catch me
                    display(['voteCounts fail in IBCC ' me.message]);
                end
            end
        end

        function savePi(obj, filenameTemplate, nIt, testId)
            Pi = exp(obj.lnPi);                        
            save(sprintf(filenameTemplate, testId, nIt), 'Pi');
        end       
    end    
end

