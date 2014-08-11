classdef CIbccVb < combiners.bcc.IbccVb
    properties
        useNegFeatures = true; %count non-occurrences of features as a ngeative response        
        negC = [];
    end
    
    methods (Static)
        function id = getId()
            id = 'Cont. IBCC-VB';    
        end
    end
    
    methods 
        function obj = CIbccVb(bccSettings, nAgents, targets, nClasses, nScores)
            obj@combiners.bcc.IbccVb(bccSettings, nAgents, targets, nClasses, nScores);
            if nScores ~= 2
                display('Continuous IBCC does not work with more than 2 scores at the moment!'); 
                display('It can be used by setting useNegFeatures=false and separating each score into a separate feature.');
            end
        end
        
        function baseData = correctBaseData(obj, baseOutputs) 
            baseData = obj.convertSparseToMat(baseOutputs);
        end           
        
        function prepareC(obj, post)
            obj.C = post;            
        end        
        
        function Count = voteCounts(obj)
            Count = zeros(obj.nClasses, obj.nScores, obj.nAgents); 
            Cpos = obj.C; 
            
            for j=1:obj.nClasses
                Tj = obj.post_T(j, :);
                totalCounts = sum(Tj);
                Count(j, 2, :) = Cpos*Tj';
                Count(j, 1, :) = Count(j, 1, :) + totalCounts - Count(j, 2, :);
            end
        end     

        function [pT, logJoint] = expectedT(obj)
            logJoint = zeros(obj.nClasses, size(obj.C, 2));
                
            Cpos = obj.C;
            if obj.useNegFeatures && isempty(obj.negC)
                obj.negC = 1-Cpos;
            end
            
            for j=1:obj.nClasses      
                %Turn -1s into zeros when totting up the positive examples
                %and -1s into 1s when totting up the negative examples
                if obj.useNegFeatures
                    lnPCT = reshape(obj.lnPi(j,2,:), 1, obj.nAgents)*Cpos + ...
                        reshape(obj.lnPi(j,1,:), 1, obj.nAgents)*(obj.negC);
                else
                    lnPCT = reshape(obj.lnPi(j,2,:), 1, obj.nAgents)*Cpos;
                end
                logJoint(j, :) = obj.lnKappa(j) + lnPCT;
            end
            rescale = repmat(-min(logJoint,[],1), obj.nClasses, 1);
            pT = exp(logJoint+rescale);
            normTerm = ones(obj.nClasses, 1)*sum(pT, 1);
            pT = pT ./ normTerm;
                        
            obj.post_T = obj.Tmat;
            obj.post_T(:, obj.testIdxs) = pT(:, obj.testIdxs);
        end        
    end
    
end

