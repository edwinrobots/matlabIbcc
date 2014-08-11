classdef  IbccVb < combiners.bcc.Ibcc    

    methods (Static)
        function id = getId()
            id = 'IBCC-VB';            
        end
    end
    
    methods       
        function obj = IbccVb(bccSettings, nAgents, targets, nClasses, nScores)
            % bccSettings - a structure containing hyperparam settings etc.
            % nAgents - number of base classifiers/agents to combine
            % targets - a vector containing the target class labels.
            % Training labels should have values from 1 to nClasses; test
            % labels should have value 0.
            % nClasses - number of target classes
            % nScores - number of output values from the base classifiers.
            obj@combiners.bcc.Ibcc(bccSettings, nAgents, targets, nClasses, nScores);            
            obj.initVariables();
        end
        
        function [post_T, sd_post_T, post_Alpha] = combineDecisions(obj, C) 
            % C - a cell array with 3 cells:-
            % C{1}: base classifier IDs
            % C{2}: object IDs for the objects to be classified
            % C{3}: agents'/base classifier output scores. 
            obj.prepareC(C);
            obj.reinitAlpha();           
            obj.iterate();                

            if obj.settings.debug
                obj.printPiEvolution();
            end            
            if obj.nClasses==2 && size(obj.post_T,1)==2
                obj.post_T = obj.post_T(2, :);
            end
            post_T = obj.post_T;
            if isempty(obj.sd_post_T)
                obj.sd_post_T = post_T .* (1-post_T); 
            end
            sd_post_T = obj.sd_post_T;
            post_Alpha = obj.Alpha;            
        end   
        
        function initVariables(obj)
            if obj.debug
                display('IBCC-VB init. Setting all variables to priors.');
            end
            obj.initLnKappa();
            obj.initET();
            obj.initLnPi();
        end        
        
        function initET(obj)      
            if isempty(obj.post_T)
                if ~isempty(obj.nObjects)
                    obj.post_T = zeros(obj.nClasses, obj.nObjects);
                else
                    obj.post_T = zeros(obj.nClasses, obj.nObjects);
                end
                obj.post_T = obj.post_T + 1/obj.nClasses;
                for j=1:obj.nClasses
                    obj.post_T(:, obj.targets==j) = 0;
                    obj.post_T(j, obj.targets==j) = 1;
                end 
            end
        end
                
        function initLnKappa(obj) 
            
            if size(obj.Nu0,2)>size(obj.Nu0,1)
                obj.Nu0 = obj.Nu0';
            end
            
            if isempty(obj.lnKappa)
                obj.lnKappa = psi(obj.Nu0) - psi(sum(obj.Nu0));
            end
        end
        
        function initLnPi(obj)           
            if isempty(obj.lnPi)
                obj.lnPi = psi(obj.Alpha0) - psi(repmat(sum(obj.Alpha0,2),[1,obj.nScores,1]));         
            end
        end        
               
        function reinitAlpha(obj)
            oldNAgents = size(obj.Alpha0,3);
            obj.nAgents = max(obj.C{1});
            if oldNAgents < obj.nAgents
                obj.setAlphaPrior(obj.Alpha0, true, obj.nAgents);
                newAlphas = repmat(obj.Alpha0(:,:,1), [1,1,oldNAgents-obj.nAgents]); 
                obj.Alpha = cat(3, obj.Alpha, newAlphas);
                obj.lnPi = psi(obj.Alpha) - repmat(psi(sum(obj.Alpha,2)), [1,obj.nScores,1]);
            end             
        end
                 
        function iterate(obj)
            converged = false;
            cIt = 0; 
            nIt = 0;
            L = -inf;
            
            diff = 0;
            
            while ~converged   
                if obj.debug
                    display(['IBCC-VB iterations: ' num2str(nIt) ', diff: ' num2str(diff) ', lower bound: ' num2str(L)]);
                end                
                
                oldET = obj.post_T;%oldL = L;
                nIt = nIt + 1;
                
                [~, lnpCT] = obj.expectedT();
                obj.expectedLnPi();
                obj.expectedLnKappa();
                
                if obj.debug
                    oldL = L;
                    L = obj.lowerBound(lnpCT); 
                    display(['Lower bound change: ' num2str(L-oldL)]);
                    if L-oldL<0
                        display('Lower Bound Error!');
                    end
                end
                diff = sum(sum(abs(obj.post_T - oldET)));

                if abs(diff) < obj.settings.convThreshold
                    cIt = cIt + 1;
                else
                    cIt = 0;
                end
                if ((cIt > obj.settings.convIt  || nIt > obj.settings.maxIt) && obj.settings.fixedNIt==0) ...
                        || (nIt >= obj.settings.fixedNIt && obj.settings.fixedNIt > 0)
                    converged = true;
                end                
            end
            
            display([num2str(nIt) ' iterations for IBCCVB.']);            
        end
                      
        
        function printPiEvolution(obj)
            classifiers = unique(obj.C{1});
            for k=classifiers'
                display(['Alpha for classifier ' num2str(k')]);
                obj.Alpha(:,:,k)
            end            
        end
        
        function updateAlpha(obj)
            Count = obj.voteCounts();
            obj.Alpha = obj.Alpha0 + Count;                
        end
        
        function expectedLnPi(obj)
            obj.updateAlpha();    
            obj.lnPi = psi(obj.Alpha);
            normTerm = psi(sum(obj.Alpha,2));
            normTerm = repmat(normTerm, 1, obj.nScores);
            obj.lnPi = obj.lnPi - normTerm;
        end
        
        function expectedLnKappa(obj)         
            obj.Nu = obj.Nu0;
            for j=1:obj.nClasses
                try
                    obj.Nu(j) = obj.Nu0(j) + sum(obj.post_T(j, :));
                    
                catch me
                    display(['bad Nu or T values: ' me]);
                end
            end  
            obj.lnKappa = psi(obj.Nu) - psi(sum(obj.Nu));
        end
                      
        function [ET] = flipCounts(obj, ET, ~, ~, C, ~, ~)
            invET = ET;
            invET(1,obj.testIdxs) = ET(2,obj.testIdxs);
            invET(2,obj.testIdxs) = ET(1,obj.testIdxs);          
            invLnPi = obj.expectedLnPi(invET);
            invLnP = obj.expectedLnKappa(invET);

            [invET, ~] = obj.expectedT(C, invLnP, invLnPi, size(ET,2));
            ET = invET;%ET.*0.5 + invET.*0.5;
        end   
        
        function ElnPCT = postLnJoint(obj, lnJoint)           
            ElnPCT = sum(lnJoint .* obj.post_T,1);
        end         
        
        function ElnPKappa = postLnKappa(obj)
            ElnPKappa = sum(gammaln(sum(obj.Nu0))-sum(gammaln(obj.Nu0)) + sum((obj.Nu0-1).*obj.lnKappa));
        end
        
        function ElnQKappa = QLnKappa(obj)
            ElnQKappa = sum(gammaln(sum(obj.Nu, 1))-sum(gammaln(obj.Nu),1) + sum((obj.Nu-1).*obj.lnKappa));
        end        
        
        %lower bound decreases sometimes occur: this only happens when both
        %kappa and pi are being updated - when one is fixed we don't have a problem.
        function [L, EEnergy, H] = lowerBound(obj, lnJoint)           
            
            ElnPCT = obj.postLnJoint(lnJoint);
            
            ElnPPi = gammaln(sum(obj.Alpha0, 2))-sum(gammaln(obj.Alpha0),2) + sum((obj.Alpha0-1).*obj.lnPi, 2);
            ElnPPi = sum(sum(ElnPPi, 1), 3);
            
            ElnPKappa = obj.postLnKappa();
            
            T = obj.post_T(obj.post_T~=0);
            ElnQT = sum(T .* log(T));

            ElnQPi = gammaln(sum(obj.Alpha, 2))-sum(gammaln(obj.Alpha),2) + sum((obj.Alpha-1).*obj.lnPi, 2);
            ElnQPi = sum(sum(ElnQPi));           
            
            ElnQKappa = obj.QLnKappa();
            
            if isinf(ElnPKappa) && isinf(ElnQKappa)
                ElnPKappa = 0;
                ElnQKappa = 0;
            end
            if isinf(ElnPPi) && isinf(ElnQPi)
                ElnPPi = 0;
                ElnQPi = 0;
            end            
            EEnergy = ElnPCT + ElnPPi + ElnPKappa;
            H = - ElnQT - ElnQPi - ElnQKappa;
            
            L = EEnergy + H;
        end     
    end
end
