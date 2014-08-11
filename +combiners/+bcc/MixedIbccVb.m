classdef MixedIbccVb < combiners.bcc.DynIbccVb
    %DYNIBCCVB Dynamic IBCC-VB
    %   For base classifiers that change. Uses a dynamic
    %   dirichlet-multinomial model to determine the confusion matrices
    %   after each data point.
    
    properties
        Cfeat = []; %features in matrix format
        featureIdxs = []; %static, continuous-response features
        dynIdxs = []; %dynamic, discrete-response base classifiers
        
        featCombiner = [];
        
        logJoint = [];
        featPi = [];
        featAlpha = [];
    end
    
    methods(Static)
        function id = getId()
            id = 'IBCC-VB + cont. features';   
        end   
    end
    
    methods %bccSettings, nAgents, targets, nClasses, nScores
        function obj = MixedIbccVb(bccSettings, nFeat, featSettings, features, nAgents, targets, nClasses, nScores)
            obj@combiners.bcc.DynIbccVb(bccSettings, nAgents, targets, nClasses, nScores);
            
            obj.featureIdxs = 1:nFeat;
            obj.Cfeat = features;
            obj.dynIdxs = nFeat:nAgents;   
            
            obj.featCombiner = combiners.bcc.CIbccVb(featSettings, nFeat, targets, nClasses, 2);
            obj.featCombiner.C = obj.Cfeat;
            obj.featCombiner.useNegFeatures = false;
            obj.featCombiner.lnKappa = zeros(1,2);
        end        
         
        function expectedLnPi(obj)
            expectedLnPi@combiners.bcc.DynIbccVb(obj, C, obj.post_T);
            
            %can we do feature selection as well as intelligent tasking?
            %This could be used to limit the size of matrices we
            %manipulate.
            obj.featCombiner.post_T = obj.post_T;
            obj.featCombiner.expectedLnPi();
            obj.featPi = obj.featCombiner.lnPi;
            obj.featAlpha = obj.featCombiner.Alpha;
        end

        function [pT, lnPT] = expectedT(obj)            
            nResp = length(obj.C{1});
            indx = sub2ind([obj.nScores length(nonZeroScores)], ...
                            obj.C{3}, ...
                            nonZeroScores);     
            lnPT = zeros(obj.nClasses, obj.nObjects);
            
            lnPi = obj.lnPi;
            lnK = obj.lnKappa;
            objs = obj.C{2}';
            for j=1:obj.nClasses                
                lnPT(j,:) = sparse(ones(1,nResp), objs, lnPi(j, indx), 1, obj.nObjects);
                if ~obj.settings.useLikelihood
                    lnPT(j,:) = lnPT(j,:) + lnK(j);
                end
            end
            
            [~, lnPFT] = obj.featCombiner.expectedT();
            lnPT = lnPT + lnPFT;
            
            rescale = repmat(-min(lnPT), obj.nClasses, 1);
            expA = exp(lnPT+rescale);%exp(lnPT(:,realIdxs)+rescale);
            
            obj.logJoint = lnPT;

            expB = repmat(sum(expA,1), obj.nClasses, 1);
            
            %stop any extreme values from causing Na
            expB(expA==Inf) = 1;
            expA(expA==Inf) = 1;
            
            pT = expA./expB;
                      
            if length(obj.targets)<1
                obj.post_T = pT;  
            else
                obj.post_T = obj.Tmat;
                obj.post_T(:, obj.testIdxs) = pT(:, obj.testIdxs);
            end
        end
    end 
end

