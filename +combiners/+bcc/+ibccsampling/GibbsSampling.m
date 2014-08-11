classdef GibbsSampling < handle 
    properties (Constant)
        % number of iterations - not needed?
        %gibbsIterations = 500;
        %number of accepted samples
        sampleInterval = 10;%10; %1; %3
        burnIn = 10;%100;%500; %20; %40
        gibbsSamples = 50;%20000; %100; %200
    end
    properties
        
        combiner;
        
        nClassifiers
        nClasses
        nScores
        
        lambda
        nu
        
        members = []
        ars
        
        initAlphaFromMeans = true;
        
        convThreshold = 0;%10^-4;
        ignoreConv = true;
    end
    
    methods
        
        function obj = GibbsSampling(combiner, nClassifiers, members)
            obj.combiner = combiner;
            
            obj.nClassifiers = nClassifiers;
            obj.nClasses = combiner.nClasses;
            obj.nScores = combiner.nScores;
            obj.lambda = combiner.lambda;
            obj.nu = obj.combiner.Nu0;
            
            if nargin > 2
                obj.members = members;
            end
            
            obj.ars = combiners.bcc.ibccsampling.ARS(obj.nClassifiers, ...
                obj.nClasses, obj.nScores, obj.lambda, obj.nu); 
        end
        
        function [Alpha0, P, T, Tknown] = initVariables(obj, Tknown, nSamples, C)
            %set initial values 
            %start with means of priors or random draws?
            Cknown = C;
            Cknown{1} = C{1}(ismember(C{2},find(Tknown)));
            Cknown{2} = C{2}(ismember(C{2},find(Tknown)));
            Cknown{3} = C{3}(ismember(C{2},find(Tknown)));
            ET = sparse(Tknown(Tknown~=0), find(Tknown~=0), 1, obj.nClasses, nSamples);
            confCounts = obj.combiner.voteCounts(ET);
            
            obj.combiner.setAlphaPrior(obj.combiner.Alpha0, true, obj.combiner.nAgents);
            Alpha0 = obj.combiner.Alpha0;
            if obj.initAlphaFromMeans            
                Alpha = obj.combiner.Alpha0 + confCounts;
                P = (obj.nu + sum(ET,2)') ./ (sum(obj.nu) + sum(sum(ET,2),1)); 
            else
                Alpha = obj.combiner.Alpha0 + confCounts;
                P = obj.combiner.sampleKappaPrior();
            end
            
            if sum(obj.nu) == 0
                P = [0.5 0.5];
            end            
            
            % no longer necessary as we expect labels in this form now: Tknown = Tknown + 1;
            Conf = Alpha ./ repmat(sum(Alpha,2), 1, size(Alpha,2));
            T = obj.combiner.sampleT(C, Conf, P, obj.members);
        end
        
        function [expectedPost, expectedConf, Alpha] = sample(obj, C, Tknown)
            nSamples = length(Tknown);
            
            [Alpha0, P, T, Tknown] = obj.initVariables(Tknown, nSamples, C);
            
            initialT = T;
            
            Alpha = Alpha0;
            currentInterval = obj.sampleInterval;
            nSamplesCollected = 0;
            
            expectedPost = zeros(1, nSamples);
            expectedConf = zeros(obj.nClasses, obj.nScores, obj.nClassifiers);
            modelEvidence = 0;
            
            i = 0;           
            currentET = 0.5;

            flipped = true; display('no flipping allowed');

            conv =obj.gibbsSamples;
            while nSamplesCollected < obj.gibbsSamples && i < 5*(obj.gibbsSamples*obj.sampleInterval+obj.burnIn)
                i = i+1;

                ET = sparse(T, (1:nSamples), 1, obj.nClasses, nSamples);
                    
                confCounts = obj.combiner.voteCounts(C, ET);          
                Conf = obj.combiner.sampleConf(Alpha + confCounts, C);
                               
                newAlpha = obj.sampleAlpha(Conf, Alpha); 
                if ~isempty(newAlpha)
                    Alpha = newAlpha;
                else
                    continue;
                end

                T = obj.combiner.sampleT(C, Conf, P, obj.members);
                
%                 if nSamplesCollected>=obj.gibbsSamples/2 && ~flipped
%                     %flip
%                     
%                     savedPost = expectedPost;
%                     savedConf = expectedConf;
%                     savedAlpha = Alpha;
%                     savedModelEvidence = modelEvidence;
%                     
% %                     post = savedPost ./ nSamplesCollected;
% %                      log(1-savedPost)
% %                     post = post(post>0 & post < 1);
% %                     logPjoint = sum(post.*log(post),2) + sum((1-post).*log(1-post),2);
%                     
%                     expectedPost = expectedPost .* 0;
%                     expectedConf = expectedConf .* 0;
%                     modelEvidence = 0;
%                     
%                     if abs(initialT - T) < 0.5*(length(T)-sum(Tknown>0)) 
%                         currentInterval = -obj.burnIn;
%                         T = T+1;
%                         T(T>obj.nClasses) = 1;
%                         T(Tknown>0) =  Tknown(Tknown>0);
%                     end %else already happened naturally
%                     flipped = true;
%                 end
                
                P = obj.sampleP(T); %also uses nu as a prior
                
                oldET = currentET;                
                
                if i>obj.burnIn && currentInterval >= obj.sampleInterval
                                        
                    if nSamplesCollected==0
                        expectedPost = zeros(1, nSamples);
                    end
                    
                    %T records labels as 1 or 2
                    expectedPost = expectedPost + T - 1;
                    expectedConf = expectedConf + Conf;
                    %expectedP = expectedP + P;
                    currentInterval = 1;
                    nSamplesCollected = nSamplesCollected + 1;
                    %display(sprintf('gibbs sample - %d', nSamplesCollected));
                    
                    currentET = expectedPost ./ nSamplesCollected;
                    if ~isempty(obj.combiner.progressMonitor)
                        obj.combiner.progressMonitor.combinerIteration(...
                            currentET, obj.combiner.getId(), i, ...
                            obj.combiner.nKnown); 
                    end
                    
                    %to calculate model evidence sum over (integrate) the
                    %joint probability
%                     lnModEvIt = sum(log(P(Tknown(Tknown>0)))) + sum(sum(sum(log(Conf).*confCounts)));
%                         + sum(log(P).*(obj.combiner.Nu0-1), 2) ...
%                         + sum(sum(sum(log(Conf).*(obj.combiner.Alpha0-1))));
%                     modelEvidence = modelEvidence + exp(lnModEvIt);
%                     display(['Model evidence: ' num2str(modelEvidence)]);
                else
                    %expectedPost = expectedPost + T - 1;
                    
                    currentInterval = currentInterval + 1;
                end               
                
                change = sum(abs(currentET - oldET));
                if ~obj.ignoreConv && change < obj.convThreshold && conv > i
		  
                    conv = nSamplesCollected;
                elseif change >= obj.convThreshold
                    conv = obj.gibbsSamples;
                end
                
                %if no change for 20 samples then assume convergence has
                %been reached
                if conv < nSamplesCollected-(20/obj.sampleInterval) && i> obj.burnIn
                    display(['breaking' num2str(conv)]);
                    break;
                end
                
                if nSamplesCollected==obj.gibbsSamples
                    display('reached the limit');
                end
            end           
            
            if exist('savedModelEvidence','var')
                nSamplesCollected = nSamplesCollected ./ 2;
            end
            
            if nSamplesCollected < obj.gibbsSamples
                display(sprintf('Collected %i.', nSamplesCollected));
            end
                        
            post = expectedPost ./ nSamplesCollected;
            post = post(post>0 & post < 1);
%             logPjoint2 = sum(post.*log(post),2) + sum((1-post).*log(1-post),2);
            
            if exist('savedModelEvidence','var')
                display([num2str(savedModelEvidence) ', ' num2str(modelEvidence)]);
            
                if savedModelEvidence > modelEvidence
                    expectedPost = savedPost;
                    expectedConf = savedConf;
                    Alpha = savedAlpha;
                else
                    display('Flipping...');
                end
            end
            
            %display(sumET./(size(T,2)*20)); %mean difference per data point over the last 20 iterations - was it still  changing significantly?
            expectedPost = expectedPost ./ nSamplesCollected;
            expectedConf = expectedConf ./ nSamplesCollected;
            %expectedP = expectedP ./ nSamplesCollected;
            
            %expectedPost = obj.flipIfNeeded(expectedPost, expectedConf, expectedP, C, Tknown, Alpha0);            
        end  

        function EPost = flipIfNeeded(obj, EPost, EPi, EP, C, Tknowns, Alpha)
            [ET, pT] = obj.expectedT(Tknowns, C, EP, EPi);
            logPjoint = sum( log(sum(ET(:, 1:length(Tknowns)) .* pT(:, 1:length(Tknowns)), 1) ), 2)...
                + sum(sum(log(obj.combiner.dirpdf(EPi, Alpha)), 1), 3)...
                + log(obj.combiner.dirpdf(EP, obj.nu));
            
            %flip everything
            EPf = EP;
            EPif = EPi;
            for j=1:obj.nClasses
                
                EPf(j) = EP(mod(j, obj.nClasses)+1);
                EPif(:, j) = EPi(:, mod(j, obj.nClasses)+1);
            end
            
            [ETf, pTf] = obj.expectedT(Tknowns, C, EPf, EPif);
            logPjointf = sum( log(sum(ETf(:, 1:length(Tknowns)) .* pTf(:, 1:length(Tknowns)), 1) ), 2)...
                + sum(sum(log(obj.combiner.dirpdf(EPif, Alpha)), 1), 3)...
                + log(obj.combiner.dirpdf(EPf, obj.nu));
            
            if logPjointf >= logPjoint
                EPost = 1 - EPost;
            end
        end
        
        function [ET, pT] = expectedT(obj, Tknowns, C, lnP, lnPi)
            logJoint = zeros(obj.nClasses, size(C, 2));
            pT = zeros(obj.nClasses, size(C, 2));
            
            [I, J, nonZeroC] = find(C);
            nonZeroIdx = sub2ind(size(C), I, J);
            
            if size(I, 2) > size(I, 1)
                I = I';
            end
            if size(nonZeroC, 2) > size(nonZeroC, 1)
                nonZeroC = nonZeroC';
            end
            for j=1:obj.nClasses
                lnPCT = zeros(1, size(C,2));
                for k=1:obj.nClassifiers
                    lnPCT = lnPCT + lnPi(j, C(k,:), k);
                end

                logJoint(j, :) = lnP(j) + lnPCT;
            end
            pT = exp(logJoint);
            normTerm = ones(obj.nClasses, 1)*sum(pT, 1);
            pT = pT ./ normTerm;
            
            ET = pT;
            ET(:, Tknowns~=0) = 0;
            ET(sub2ind(size(pT), Tknowns(Tknowns~=0), find(Tknowns~=0))) = 1;
        end                
        
        function Alpha = sampleAlpha(obj, Conf, Alpha)
            if obj.combiner.fixedAlpha
                Alpha = obj.combiner.Alpha0;
            else
                Alpha = obj.ars.sampleAlpha(Conf, Alpha);
            end            
        end
        
%         function confCounts = confusionCounts(obj, T, C, nSamples)
%             confusion counts: the observations/predictions produced by 
%             the current model giving the observed confusion counts            
%             confCounts are the counts for each classifier output
%             given each true label                        
%             confCounts = ones(obj.nClasses, obj.nScores, obj.nClassifiers);       
%             
%             T = ones(obj.nClassifiers, 1) * T;
%             
%             %assuming 2 classes
%             for j=1:obj.nClasses
%                 for s=1:obj.nScores
%                     if obj.combiner.scoreSet
%                         confCounts(j, s, :) = sum((C{3}.*(T(C{2})==j)) ==s, 2);                        
%                     else 
%                         confCounts(j, s, :) = sum((C.*(T==j)) ==s, 2);
%                     end
%                 end
%             end
%         end        
        
        function Psample = sampleP(obj, T)
            % p(P|Alpha, conf, T, C) = p(P| T) = p(T|P)p(P)/p(T) prop. to 
            % p(T|P)dirichlet(nu) = P(T=Tsample)dirichlet(nu) =
            % dirichlet(T+nu)
            params = obj.nu;
            for c = 1:obj.nClasses
                params(c) = params(c) + sum(T==c);
            end
                        
            scaleFactors = ones(1, obj.nClasses);
            
            %repeat this a few times so we can see whether it's correct.
            if length(T) > 10
                nPSamples = 10;
            else
                nPSamples = 1;
            end
            
            for i=1:nPSamples
                Psample = gamrnd(params, scaleFactors);
                Psample = Psample ./ sum(Psample);
            end
        end
    end
end