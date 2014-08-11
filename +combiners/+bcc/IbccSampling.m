classdef IbccSampling < combiners.bcc.Ibcc
    %IBCCSAMPLING IBCC class that uses sampling
    %   Detailed explanation goes here
    
    properties
        lambda = [0.5 1; 1 0.5];%[0.5 1; 1 0.5];%[0.4 1; 1 0.4]; %[1 1; 1 1];
        lambdaMag; lambdaSym;  
        
        %maximum number of times we will run the gibbs sampler before
        %giving up
        maxSampleAttempts = 20;
        
        samplingType = 'Gibbs'; %or 'MH';      
        minConf = 10^-6;              
    end
    
    methods (Static)
        function sl = shortLabel
            sl = 'IbccSampling';
        end
    end
     
    methods
        
        function obj = IbccSampling(bccSettings, nAgents, K, targets, agents, nClasses, nScores)
            obj@combiners.bcc.Ibcc(bccSettings, nAgents, K, targets, agents, nClasses, nScores);
                        
            obj.setTrustedAgents();
                        
            obj.nonDet = true; %non-deterministic
        end        
                
        function setTrustedAgents(obj, trustFinalAgent)
            if nargin > 1
                obj.trustFinalAgent = trustFinalAgent;
            end
            for trusted=obj.trustFinalAgent
                trustedStart = obj.nAgents - length(obj.trustFinalAgent);
                if isempty(obj.trustedLambda)
                    obj.lambda(:,:,trustedStart+trusted) = obj.lambdaMag(1+trusted)*obj.lambdaSym(1+trusted);
                    idxs = sub2ind(size(obj.lambda), ...
                        1:obj.nClasses, 1:obj.nClasses, ones(1,obj.nClasses)*trustedStart+trusted);
                    obj.lambda(idxs) = obj.lambdaMag(1+trusted)*(1-obj.lambdaSym(1+trusted));
                else
                    obj.lambda(:, :, trustedStart+trusted) = obj.trustedLambda(:,:,trusted);
                end
            end
        end 
        
                
        function combinedPost = combineCluster(obj, post, T, members)          
           
            obj.nKnown = round(obj.nKnown);
            Tknown = T;
            
            C = obj.prepareC(post);
                          
            %gibbs generates a bunch of samples, and takes the mean as
            %the expected value of the distribution.
            if strcmp(obj.samplingType, 'Gibbs')
                if nargin > 3
                    gibbsSampler = combiners.bcc.ibccsampling.GibbsSampling(obj, obj.nAgents, members);
                else
                    gibbsSampler = combiners.bcc.ibccsampling.GibbsSampling(obj, obj.nAgents);
                end
                expectedT = nan;
                i = 1;
                while isnan(expectedT)
                    if i > obj.maxSampleAttempts
                        break
                    end
                    [expectedT, ~, expectedAlpha] = gibbsSampler.sample(C, Tknown);
                    i = i+1;
                end    
                %save the expected confusion matrix
                %dlmwrite(sprintf('%s%s_%s', obj.confSave, 'ibccgibbs',
                %obj.datasetLabel), expectedConf);
                combinedPost = expectedT;
            elseif strcmp(obj.samplingType, 'MH')
                if nargin > 3
                    mhSampler = combiners.bcc.ibccsampling.MHSampling(obj, members);
                else
                    mhSampler = combiners.bcc.ibccsampling.MHSampling(obj);
                end
                combinedPost = mhSampler.sample(C, Tknown);
            end
            
            obj.Alpha = obj.Alpha + expectedAlpha;
        end 
        
        function ConfSample = sampleConf(obj, Alpha, C)
            agents = unique(C{1});

            ConfSample = zeros(obj.nClasses, obj.nScores, obj.nAgents);
            
            if obj.nScores > 2 || obj.nClasses > 2
                for j=1:obj.nClasses
                    K = agents;
                    while ~isempty(K)
                        for c=1:obj.nScores
                            ConfSample(j, c, K) = betarnd(reshape(Alpha(j,c,K), 1, length(K)), ...
                                reshape(sum(Alpha(j,:,K),2)-Alpha(j,c,K), 1, length(K)) );

                        end
                        [J, K] = find(ConfSample(j, :, agents)<obj.minConf, 1);    
                        ConfSample(j, K) = obj.minConf;
                        K = [];
                    end
                end
            else
                for j=1:obj.nClasses
                    K = 1:obj.nAgents;
                    while ~isempty(K)
                        ConfSample(j, 1, K) = betarnd(Alpha(j, 1, K), Alpha(j, 2, K));
                        ConfSample(j, 2, K) = 1 - ConfSample(j, 1, K);
                        [J, K] = find(ConfSample(j, :, :)<obj.minConf, 1);
                        K = ceil(K ./ obj.nClasses);
                    end
                end             
            end
        end 
        
        %THIS ONLY WORKS WITH TWO CLASSES
        function [T, postTi] = sampleT(obj, C, Conf, P,  members)     
            nSamples = length(obj.targets);

            if nargin > 5 && ~isempty(members)
                display('breakage breakage breakage');
                postTi = obj.posteriorTi(C, Conf, P, nSamples, members);
            else
                postTi = obj.posteriorTi(C, Conf, P, nSamples);
            end
            
            T = zeros(1, length(postTi));

            validT = unique(C{2});
 
            randoms = rand(1, length(validT));
            
            T(validT) = (randoms<postTi(validT)) + 1;
            
            %display('breaking tknowns again');
            T(obj.targets~=0) = obj.targets(obj.targets~=0);
        end   
        
        
        function kappa = sampleKappaPrior(obj)
            %class proportions. 
            kappa = gamrnd(obj.Nu0, ones(1, obj.nClasses));
            kappa = kappa ./ (sum(kappa)*ones(1, obj.nClasses));
        end        

        %Find the p(Ti | Ci, P, Conf, Alpha) =  p(Ti | Ci, P, Conf)
        function postTi = posteriorTi(obj, Conf, P, nSamples)
            
            pCiTi = zeros(1, nSamples, obj.nClasses);
            for j=1:obj.nClasses
                validDecs = obj.C{3};
                validA = obj.C{1};
                validN = obj.C{2};
                
                indxs = sub2ind([size(Conf,2), size(Conf,3)], validDecs, validA);
                logPiValues = log(Conf(j,indxs));
                pCi_giv_Ti_eqj = sparse(validA, validN, logPiValues, obj.nAgents, nSamples);
                pCi_giv_Ti_eqj = exp(sum(pCi_giv_Ti_eqj, 1));
                
                %rows in pCi_given_Ti correspond to true labels. Columns
                %correspond to samples.                
                pCiTi(1, :, j) = prod(pCi_giv_Ti_eqj, 1) .* P(j);
            end                       
            postTi = pCiTi(:, :, 2) ./ sum(pCiTi, 3);
        end        
    end
end

