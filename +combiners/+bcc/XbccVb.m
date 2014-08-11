classdef  XbccVb < combiners.bcc.IbccVb %needs a better name!
    %Interpolating BCC (interBcc)
    %BRC (Bayesian report combination, because reporters describe surrounding areas, not just distinct data points like classifiers?)
    %Is it general/extensible enough a name? Does it work with documents,
    %for instance? What are catchy names of other algorithms? "naive
    %Bayes", "EM", "GP"... often two simple words. -- can we find a single
    %word for report combination?
    
    %Something more catchy.
    %Manifold+Classifier?
    %Maps + Events + Bayes = BEM (Bayesian Event Mapping)
    % Semi-supervised
    properties
        T_reps %The target values at the locations of the reports C
        nReps % Number of report locations.
        
        nx %grid size. May be able to generalise away from 2-D grids
        ny
        
        f_pr %priors from the GP
        Cov_pr
        s_pr
        
        f %posteriors from the GP
        Cov
        s
    end

    methods (Static)
        function id = getId()
            id = 'Extended BCC-VB';            
        end
    end
    
    methods      
        function obj = XbccVb(bccSettings, nAgents, targets, nClasses, nScores, nx, ny)
            obj@combiners.bcc.IbccVb(bccSettings, nAgents, targets, nClasses, nScores); 
            
            display('Setting up a 2-D grid. This should be generalised!');
            
            obj.nx = nx;
            obj.ny = ny;
            
            obj.nObjects = nx*ny;
            obj.lnKappa = [];
            obj.post_T = [];
            obj.initVariables();
        end    
        
        function setTargets(obj, targets)
            obj.targets = targets;
            obj.Tmat = targets;
            obj.trainIdxs = find(targets~=0);
            obj.testIdxs = find(targets==0);
        end       
        
        function prepareC(obj, Clist)
            C3 = round(Clist{3});
            display('Is the mapping thing right?');
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
            
            obj.C = Clist;
            
            obj.C{2} = sub2ind([obj.nx,obj.ny],obj.C{2}(1,:),obj.C{2}(2,:));            
           
            obj.nReps = length(unique(obj.C{2}));
        end
        
        %TODO during init
        %1. need to initialise nObjects to the number of grid points?.
        %2. Need to map each report to a separate object. Alter C to add
        %extra column
        %3. Need to match a set of objects at a location to a kappa_i.
        %Kappa should be updated using gpgrid.        
        
        function initLnKappa(obj) 
            if isempty(obj.lnKappa)
                %start with a homogeneous grid
                if size(obj.Nu0,1)<size(obj.Nu0,2)
                    obj.Nu0 = obj.Nu0';
                end
                obj.lnKappa = psi(obj.Nu0) - psi(sum(obj.Nu0));
                
                if size(obj.Nu0,2)<obj.nObjects
                    obj.Nu0 = repmat(obj.Nu0, [1, obj.nObjects]);
                end
                
                obj.lnKappa = repmat(obj.lnKappa, [1, obj.nObjects]);
                
                [obj.f_pr, obj.Cov_pr,~,~,obj.s_pr] = gpgrid({[],[],[]}, obj.nx, obj.ny);
            end
        end

        function expectedLnKappa(obj)
            reports = obj.C;
            %estimate of the true positive observations
            %Number of positive responses given positive true labels
            nPosPos = obj.T_reps(1,:).*obj.C{3}(1,:);
            %Number of negative responses given positive true labels
            nNegPos = obj.T_reps(2,:).*(obj.C{3}(2,:)-obj.C{3}(1,:));
            reports{3} = [nPosPos + nNegPos; obj.C{3}(2,:)]; 
            
            success = 0;
            while ~success
                try
                    [ obj.f, obj.Cov, mPr, sdPr, obj.s] = gpgrid(reports, obj.nx, obj.ny);
                    success = 1;
                catch err
                    display('Error with gpgrid');
                end
            end
            
            %convert to pseudo-counts
            totalNu = mPr.*(1-mPr) ./ (sdPr.^2) - 1;
            obj.Nu(2,:) = totalNu .* mPr;
            obj.Nu(1,:) = totalNu .* (1-mPr);
            
            obj.lnKappa = [log(1-mPr'); log(mPr')];  
            
            obj.sd_post_T = sdPr;
            
%             figure(1);
%             surf(reshape(mPr, obj.nx, obj.ny))
%             zlabel('mean Pr(damage response from individual)','FontSize',15);
%             xlabel('Long','FontSize',15);
%             ylabel('Lat','FontSize',15);
%             a=axis;
%             a(5)=0;
%             a(6)=1;
%             axis(a);
        end   
                
        function [pT, lnpCT] = expectedT(obj)
            lnpCT = obj.updateReports();
            pT = exp(obj.lnKappa);
                        
            if length(obj.targets)<1
                obj.post_T = pT;
            else
                obj.post_T = obj.Tmat;
                obj.post_T(:, obj.testIdxs) = pT(:, obj.testIdxs);
                obj.sd_post_T(obj.trainIdxs) = 0;
            end 
        end
        
        function lnpCT = updateReports(obj)
            lnPi = obj.lnPi;
            kIdxs = obj.C{1};
            objIndx = obj.C{2};
            
            lnK = obj.lnKappa(:,objIndx); 
            
            nPos = repmat(obj.C{3}(1,:), obj.nClasses, 1);
            nTotal = repmat(obj.C{3}(2,:), obj.nClasses, 1);
            pT_pos = reshape(lnPi(:,2,kIdxs), [obj.nClasses, length(kIdxs)])  + lnK; 
            pT_neg = reshape(lnPi(:,1,kIdxs), [obj.nClasses, length(kIdxs)])  + lnK;
            lnpCT = pT_pos.*nPos + pT_neg.*(nTotal-nPos);
            
            pT_pos = exp(pT_pos(2,:)) ./ sum(exp(pT_pos),1);            
            pT_neg = exp(pT_neg(2,:)) ./ sum(exp(pT_neg),1);            
            obj.T_reps = [pT_pos; pT_neg];
        end
        
        function Count = voteCounts(obj)
            Count = zeros(obj.nClasses, obj.nScores, obj.nAgents);    
            Tj = obj.T_reps;
            Count(2, 2, :) = obj.C{3}(1,:) .* Tj(1,:);
            Count(2, 1, :) = (obj.C{3}(2,:)-obj.C{3}(1,:)) .* Tj(2,:);
            Count(1, 2, :) = obj.C{3}(1,:) .* (1-Tj(1,:));
            Count(1, 1, :) = (obj.C{3}(2,:)-obj.C{3}(1,:)) .* (1-Tj(2,:)); 
        end
        
        function ElnPCT = postLnJoint(obj, lnJoint)
            %should we be including the last term here?
            ElnPCT = sum(sum(lnJoint.*obj.T_reps));% + sum(sum(obj.lnKappa.*obj.post_T));
        end     
        
        function ElnPKappa = postLnKappa(obj)
            %Must be replaced by a Gaussian prior by running gpgrid with no
            %training points.
%             ElnPKappa = sum(gammaln(sum(obj.Nu0))-sum(gammaln(obj.Nu0)) + sum((obj.Nu0-1).*obj.lnKappa));
            a = obj.lnKappa(2,:) - obj.lnKappa(1,:);
            p = log(mvnpdf(a'./obj.s_pr, obj.f_pr, obj.Cov_pr));
            ElnPKappa = sum(p);
            if isnan(ElnPKappa) || isinf(ElnPKappa)
                display('help');
            end
        end
        
        function ElnQKappa = QLnKappa(obj)
            %Must be replaced by a Gaussian using the outputs of gpgrid f
            %and C. Need to do a reverse sigmoid to lnKappa to get its
            %value in gp-space, then evaluate the gaussian pdf at that
            %point.
%             ElnQKappa = sum(gammaln(sum(obj.Nu, 1))-sum(gammaln(obj.Nu),1) + sum((obj.Nu-1).*obj.lnKappa));
            a = obj.lnKappa(2,:) - obj.lnKappa(1,:);
            p = log(mvnpdf(a'./obj.s, obj.f, obj.Cov));
            ElnQKappa = sum(p);     
            if isnan(ElnQKappa) || isinf(ElnQKappa)
                display('help');
            end            
        end 
        
        function [post_T, sd_post_T, post_Alpha] = combineDecisions(obj, C) 
            [post_T, sd_post_T, post_Alpha] = obj.combineDecisions@combiners.bcc.IbccVb(C);
            
            post_T = reshape(post_T, obj.nx, obj.ny);
            sd_post_T = reshape(sd_post_T, obj.nx, obj.ny);
        end
    end
end
