classdef AbstractCombiner < matlab.mixin.Copyable
    %ABSTRACTCOMBINER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties (SetAccess = protected)
        targets
        nAgents
        nObjects
        nClasses
        nScores
        debug = true %show debug/progress info or not
    end
    
    properties (GetAccess = public)
        post_T = []%posterior estimates of the set of target labels
        sd_post_T = [] %standard deviation of the target data points in the posterior
        noScore = 0 %value used to indicate a missing score. Was -1 but 0 should indicate no score here, with class labels and scores > 0
    end
    
    methods
        function obj = AbstractCombiner(nAgents, nClasses, nScores, targets)
            obj.nClasses = nClasses;
            obj.nScores = nScores;
            obj.nAgents = nAgents;            
            if exist('targets', 'var')            
                obj.setTargets(targets);
            end
        end
        
        function setTargets(obj, targets)
            obj.targets = targets;
        end

        function baseData = convertSparseToMat(obj, baseOutputs)
            if iscell(baseOutputs) && length(baseOutputs)==3
                baseData = sparse(baseOutputs{1}, baseOutputs{2}, baseOutputs{3});
                obj.nAgents = max(baseOutputs{1});
            else
                baseData = baseOutputs;
                obj.nAgents = size(baseData,1);
            end
        end
        
        function baseData = convertMatToSparse(obj, baseOutputs)
            if iscell(baseOutputs) && length(baseOutputs)==3
                baseData = baseOutputs;
                obj.nAgents = max(baseOutputs{1});
            else
                baseData = cell(1, 3);
                baseData{3} = reshape(baseOutputs, numel(baseOutputs), 1);
                baseData{1} = repmat((1:size(baseOutputs,1))', size(baseOutputs,2), 1);
                baseData{2} = repmat((1:size(baseOutputs,2))', size(baseOutputs,1), 1);
                obj.nAgents = size(baseOutputs,1);
            end
        end

        function baseData = correctBaseData(obj, baseOutputs) 
            baseData = obj.convertSparseToMat(baseOutputs);
        end        
    end
    
    methods (Abstract)
        [post_T, sd_post_T, post_Alpha] = combineDecisions(obj, baseOutputs);
    end
    
    methods (Static, Abstract)
        %A string that identifies the type of classifier
        id = getId();
    end
end

