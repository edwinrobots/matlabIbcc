classdef DynIbccVb < combiners.bcc.IbccVb
    %DYNIBCCVB Dynamic IBCC-VB
    %   For base classifiers that change. Uses a dynamic
    %   dirichlet-multinomial model to determine the confusion matrices
    %   after each data point.
    
    properties
        Wmean0; %weights - the state object; related to posterior Alphas
        P0; %covariance of the weights
        
        AlphaSum = []; 
        
        initAll = false;
        
        Wmean_filter;
        P_filter;

        q_t;
        eta_tminus;
        r_tminus;
        K_kal;
        bigLambdaBar;
        smallLambdaBar;

        Cscores;
        C3;
        C1_desparse;   
        
        qtHack = false;
    end
    
    methods(Static)
        function id = getId()
            id = 'DynIBCC-VB';    
        end
    end
    
    methods
%         bccSettings, nAgents, targets, nClasses, nScores
        function obj = DynIbccVb(bccSettings, nAgents, targets, nClasses, nScores)
            obj@combiners.bcc.IbccVb(bccSettings, nAgents, targets, nClasses, nScores);          
        end      
        
        function [Wmean0 P0] = alphaToStateDist(obj) 
            %convert Alpha to distribution over state variables W with mean
            %Wmean0 and covariance P0
            Wmean0 = log(obj.Alpha0./(repmat(sum(obj.Alpha0, 2), 1, obj.nScores)-obj.Alpha0)); %one cell entry per data point
            P0 = zeros(obj.nClasses, obj.nClasses, length(obj.C{1}), obj.nScores); %one cell for each class
            for s=1:obj.nScores
                for j=1:obj.nClasses
                    P0(j,j,:,s) = 1./obj.Alpha0(j,s,1) + 1./(sum(obj.Alpha0(j,:,1), 2)-obj.Alpha0(j,s,1));
                end
            end
        end
        
        function prepareC(obj, Clist)
            prepareC@combiners.bcc.IbccVb(obj,Clist);
            
            obj.C1_desparse = full(obj.C{1});
            obj.initAlpha(length(obj.C{1})); 
            [obj.Wmean0 obj.P0] = obj.alphaToStateDist();
        end
        
        function r_tminus = calculateR(obj, ET_n, P_tminus)
            r_tminus = zeros(1, obj.nScores);
            for s=1:obj.nScores
                r_tminus(1, s) = ET_n' * P_tminus(:,:,s) * ET_n;
            end
        end
                
        function AlphaPr = updateAlpha(obj, prior)
            
            if ~exist('prior','var')
                prior = false;
            end
            
            %Choose an implementation: either multi-class or simpler 
            %implementation for binary classification problem
            if obj.nScores > 2
                AlphaPr = obj.updateAlphaMultiClass(prior);
            else
                AlphaPr = obj.updateAlphaBinary(prior);
            end         
        end
        
        function expectedLnPi(obj)
            %specifying agentIdx means we only update for that agent from
            %the last score in C and use the given counts for others
            obj.updateAlpha();
            obj.lnPi = psi(obj.Alpha) - psi(obj.AlphaSum);
        end        
            
        function AlphaPr = updateAlphaBinary(obj, prior)
            
            %Initialise variables
            nResp = length(obj.C{1});
            nClassifiers = length(obj.C1_desparse); 
            
            if isempty(obj.Alpha)
                obj.Alpha = zeros(obj.nClasses, obj.nScores, nResp);
                obj.AlphaSum = zeros(obj.nClasses, obj.nScores, nResp); %each score is a pseudocount from a different total!           
            end
            if prior
                AlphaPr = zeros(obj.nScores,nResp);
            else
                AlphaPr = [];
            end

            startFrom = 1;
            backTo = 1;
            prevObs = zeros(1, nClassifiers);    
            subsObs = zeros(1, nClassifiers);
            
            nNewPoints = nResp - size(obj.q_t,2);
            
            obj.q_t = [obj.q_t zeros(1,nNewPoints)];
            obj.eta_tminus = [obj.eta_tminus; zeros(nNewPoints, 1)];
            obj.r_tminus = [obj.r_tminus; zeros(nNewPoints, 1)];
            r_tpost = zeros(nResp, 1);
            obj.K_kal = zeros(obj.nClasses, nResp);
            
            I = eye(obj.nClasses, obj.nClasses);
            
            %previous observation for each classifier. 0 means no observations
            Wmean_post = zeros(obj.nClasses, nResp);
            P_post = zeros(obj.nClasses, obj.nClasses, nResp);
            
            eta_tpost = zeros(nResp, 1);

            R = zeros(nResp, 1);
                        
            reshapedP0 = reshape(obj.P0(:,:,1,1), obj.nClasses, obj.nClasses);            
            
            %FORWARD PASS
            for n=startFrom:nResp      
                                
                k = obj.C1_desparse(n);
                tminus = prevObs(k);
                
                c = obj.C{3}(n);
                i = obj.C{2}(n);
                h = obj.post_T(:,i);

                if tminus==0
                    Wmean_tminus = obj.Wmean0(:,1,n);
                    P_tminus = reshapedP0;
                    q_tminus = 0;
                else
                    Wmean_tminus = Wmean_post(:,tminus);
                    P_tminus = P_post(:,:,tminus);
                    q_tminus = obj.q_t(tminus);
                end    
                
                eta_tminus_n = h' * Wmean_tminus;

                if obj.qtHack>0
                    P_tminus = P_tminus + obj.settings.changeRateMod*obj.qtHack*I;
                else
                    P_tminus = P_tminus + obj.settings.changeRateMod*q_tminus*I;
                end
                r_tminus_n = h' * P_tminus * h;                

                k_t = (1./r_tminus_n) .* (1+exp(eta_tminus_n));
                if prior
                    AlphaPr(:,n) = k_t;
                end
                
                obj.eta_tminus(n,:) = eta_tminus_n; 
                obj.r_tminus(n, :) = r_tminus_n;

                m_t = k_t .* (1+exp(-eta_tminus_n));

                y = 0; 
                if c==1
                    y = 1;
                elseif c==0 %unknown
                    y = k_t ./ m_t;
                end 
                
                %estimate posteriors of eta and r
                k_tpost = k_t + y;
                denom = m_t-k_t+1-y;
                eta_tpost(n) = log(k_tpost ./ denom);
                r_tpost(n) = (1./k_tpost) + (1./denom);
                z = (eta_tpost(n) - eta_tminus_n);
                
                u_tpost = (k_tpost/(m_t+1))*(1-(k_tpost/(m_t+1)));
                u_tminus  = (k_t/m_t)*(1-(k_t/m_t));

                obj.q_t(n) = (u_tpost>u_tminus) * (u_tpost-u_tminus);%adding the uncertainty in pi as a substitute for the uncertainty in the label, since this model is about predicting pi

                corrections = isinf(r_tpost(n));
                if corrections
                    r_tpost(n) = r_tminus_n;
                end
                
                K_kal_n = P_tminus' * h ./ r_tminus_n;
                obj.K_kal(:,n) = K_kal_n;
                
                R(n) = 1-(r_tpost(n)./r_tminus_n);   

                %calculate weight updates
                
                Wmean_post(:,n) = Wmean_tminus + K_kal_n .*z;                    
                    
                Kh = K_kal_n * h' ;
                P_post(:,:,n) = P_tminus - (Kh * P_tminus .* R(n));
                
                prevObs(k) = n;
            end
            
            if prior
                return;
            end
 
            %used to be nClassifiers, could use a lot of memory
            obj.bigLambdaBar = zeros(obj.nClasses, obj.nClasses, nResp);
            obj.smallLambdaBar = zeros(obj.nClasses, nResp);
                            
            %BACKWARD PASS               
            for n=fliplr(backTo:nResp)
                
                k = obj.C1_desparse(n);
                tplus = subsObs(k);
                subsObs(k) = n;
                
                i = obj.C{2}(n);
                h = obj.post_T(:,i);

                if tplus==0
                    bigLambdaHat = zeros(obj.nClasses, obj.nClasses);
                    smallLambdaHat = zeros(obj.nClasses, 1);
                else
                    bigLambdaHat = obj.bigLambdaBar(:,:,tplus);
                    smallLambdaHat = obj.smallLambdaBar(:,tplus);
                end
                delta_Wmean = P_post(:,:,n)*smallLambdaHat;
                delta_Ppost = P_post(:,:,n)*bigLambdaHat*(P_post(:,:,n)');

                 Wmean_post(:,n) = Wmean_post(:,n) - delta_Wmean;                
                 P_post(:,:,n) = P_post(:,:,n) - delta_Ppost;

%                 display('updating controversially early in smoother - potentially over-smoothing');
                eta_tpost(n) = h' * Wmean_post(:, n);                
                r_tpost(n) = h' * P_post(:,:,n) * h;      
                
                invA = 1./obj.r_tminus(n);
                
                z = eta_tpost(n) - obj.eta_tminus(n);               
                R(n) = 1-(r_tpost(n)./obj.r_tminus(n));
                
                hinvAR = h*(invA .* R(n));           
                hinvAz = h* (z .* invA);
                
                B = I - obj.K_kal(:,n)*h';                
                obj.smallLambdaBar(:,n) = -hinvAz + B'*smallLambdaHat;
                obj.bigLambdaBar(:,:,n) = hinvAR*h' + B'*bigLambdaHat*B;

                %Values for the VB updates: calculate the alphas
                %each class by setting inputs to 1 for each class
                r_tvb = diag(P_post(:,:,n));

                k_tvb = (1./r_tvb) .* (1+exp(Wmean_post(:,n)));
                m_tvb = k_tvb .* (1+exp(-Wmean_post(:,n)));
  
                obj.Alpha(:, 1, n) = k_tvb;
                obj.AlphaSum(:, 1, n) = m_tvb;
            end
            obj.Alpha(:, 2, :) = obj.AlphaSum(:, 1, :) - obj.Alpha(:, 1, :);
            obj.AlphaSum(:, 2, :) = obj.AlphaSum(:, 1, :);
                        
            obj.C3 = obj.C{3};
        end
               
        function AlphaPr = updateAlphaMultiClass(obj, prior)
            
            nResp = length(obj.C{1});
            if isempty(obj.Alpha)
                obj.Alpha = zeros(obj.nClasses, obj.nScores, nResp);
                obj.AlphaSum = zeros(obj.nClasses, obj.nScores, nResp); %each score is a pseudocount from a different total!             
            end
            if prior
                AlphaPr = zeros(obj.nScores,nResp);
            else
                AlphaPr = [];                
            end

            nClassifiers = length(obj.C1_desparse);             
       
            startFrom = 1;
            backTo = 1;
            prevObs = zeros(1, nClassifiers);    
            subsObs = zeros(1, nClassifiers);
            
            nNewPoints = nResp - size(obj.q_t,2);
            
            obj.q_t = [obj.q_t zeros(obj.nScores,nNewPoints)];
            obj.eta_tminus = [obj.eta_tminus; zeros(nNewPoints, obj.nScores)];
            obj.r_tminus = [obj.r_tminus; zeros(nNewPoints, obj.nScores)];
            r_tpost = zeros(nResp, obj.nScores);
            obj.K_kal = zeros(obj.nClasses, obj.nScores, nResp);
            
            I = eye(obj.nClasses, obj.nClasses);
            
            %previous observation for each classifier. 0 means no observations
            Wmean_post = zeros(obj.nClasses, obj.nScores, nResp);
            P_post = zeros(obj.nClasses, obj.nClasses, obj.nScores, nResp);
            
            eta_tpost = zeros(nResp, obj.nScores);

            R = zeros(nResp, obj.nScores);
                                                
            %FORWARD PASS
            for n=startFrom:nResp      
                
                k = obj.C1_desparse(n);
                tminus = prevObs(k);
                
                c = obj.C{3}(n);
                i = obj.C{2}(n);
                h = obj.post_T(:,i);

                if tminus==0
                    Wmean_tminus = obj.Wmean0(:,:,n);
                    P_tminus = reshape(obj.P0(:,:,n,:), obj.nClasses, obj.nClasses, obj.nScores);
                    q_tminus = zeros(obj.nScores,1);
                else
                    Wmean_tminus = Wmean_post(:,:,tminus);
                    P_tminus = P_post(:,:,:,tminus);
                    q_tminus = obj.q_t(:, tminus);
                end    
                
                eta_tminus_n = h' * Wmean_tminus;
                              
                for s=1:obj.nScores
                    P_tminus(:,:,s) = P_tminus(:,:,s) + obj.settings.changeRateMod.*q_tminus(s) * I;
                end
                P_tminus_r = reshape(P_tminus(:,:,:), obj.nClasses, obj.nClasses*obj.nScores);
                r_tminus_n = h' * reshape(h'*P_tminus_r, obj.nClasses, obj.nScores);                

                k_t = (1./r_tminus_n) .* (1+exp(eta_tminus_n));
                if prior
                    AlphaPr(:,n) = k_t; 
                end

                obj.eta_tminus(n,:) = eta_tminus_n; 
                obj.r_tminus(n, :) = r_tminus_n;

                m_t = k_t .* (1+exp(-eta_tminus_n));

                y = zeros(1, obj.nScores); 
                if c==0 %unknown
                    y = k_t ./ m_t;
                else
                    y(c) = 1;
                end 
                
                %estimate posteriors of eta and r
                k_tpost = k_t + y;
                denom = m_t-k_t+1-y;
                eta_tpost(n,:) = log(k_tpost ./ denom);
                r_tpost(n,:) = (1./k_tpost) + (1./denom);

                u_tpost = k_tpost.*(m_t+1-k_tpost) ./ (m_t+1).^2;
                u_tminus = k_t.*(m_t-k_t) ./ m_t.^2;
                
                obj.q_t(:,n) = ((u_tpost>u_tminus) .* (u_tpost-u_tminus))';
                if c==0 
                    labelUncertainty = y.*(1-y);
                    obj.q_t(:, n) = obj.q_t(:, n) + labelUncertainty;      
                end

                corrections = isinf(r_tpost(n,:));
                r_tpost(n,corrections) = r_tminus_n(corrections);
                
                tmp = reshape(P_tminus(:,:,:), obj.nClasses, obj.nClasses*obj.nScores)' * h;
                K_kal_n = reshape(tmp, [obj.nClasses, obj.nScores]);
                
                R(n,:) = 1-(r_tpost(n,:)./r_tminus_n);   

                %calculate weight updates
                for s=1:obj.nScores
                    
                    Wmean_post(:,s,n) = Wmean_tminus(:,s) + K_kal_n(:,s).*...
                        (eta_tpost(n,s) - eta_tminus_n(s))./r_tminus_n(s);                    
                    
                    obj.K_kal(:,s,n) = K_kal_n(:,s)./r_tminus_n(s);
                    Kh =  obj.K_kal(:,s,n) * h' ;
                    P_post(:,:,s,n) = P_tminus(:,:,s) - (Kh * ... 
                        P_tminus(:,:,s) .* R(n,s));
                end 
                
                prevObs(k) = n;
            end
            
            if prior
                return
            end
 
            %used to be nClassifiers, could use a lot of memory
            obj.bigLambdaBar = zeros(obj.nClasses, obj.nClasses, obj.nScores, nResp);
            obj.smallLambdaBar = zeros(obj.nClasses, obj.nScores, nResp);
            
            delta_Wmean = zeros(obj.nClasses, obj.nScores);
            delta_Ppost = zeros(obj.nClasses, obj.nClasses, obj.nScores);

            r_tvb = zeros(obj.nClasses, obj.nScores);
                
            %BACKWARD PASS               
            for n=fliplr(backTo:nResp)
                
                k = obj.C1_desparse(n);
                tplus = subsObs(k);
                subsObs(k) = n;
                
                i = obj.C{2}(n);
                h = obj.post_T(:,i);

                if tplus==0
                    bigLambdaHat = zeros(obj.nClasses, obj.nClasses, obj.nScores);
                    smallLambdaHat = zeros(obj.nClasses, obj.nScores);
                else
                    bigLambdaHat = obj.bigLambdaBar(:,:,:,tplus);
                    smallLambdaHat = obj.smallLambdaBar(:,:,tplus);
                end
                for s=1:obj.nScores  
                    delta_Wmean(:,s) = P_post(:,:,s,n)*smallLambdaHat(:, s);
                    delta_Ppost(:,:,s) = P_post(:,:,s,n)*bigLambdaHat(:,:,s)*(P_post(:,:,s,n)');
                end

%                 r_tN = zeros(1,obj.nScores);                   
                Wmean_post(:,:,n) = Wmean_post(:,:,n) - delta_Wmean;                
                P_post(:,:,:,n) = P_post(:,:,:,n) - delta_Ppost;
                
                eta_tpost(n,:) = h' * Wmean_post(:,:,n);
                
                B = zeros(obj.nClasses, obj.nClasses, obj.nScores);
                for s=1:obj.nScores                                                
                    B(:,:,s) = I - obj.K_kal(:,s,n) *h';                
                    r_tpost(n, s) = h' * P_post(:,:,s,n) * h;
                end
                
                invA = 1./obj.r_tminus(n,:);
                
                z = eta_tpost(n,:) - obj.eta_tminus(n,:);
                hinvAz = h* (z .* invA);
                
                R(n,:) = 1-(r_tpost(n,:)./obj.r_tminus(n,:));
                hinvAR = h*(invA .* R(n,:)); 
                
                for s=1:obj.nScores
                    obj.smallLambdaBar(:,s,n) = -hinvAz(:,s) + B(:,:,s)'*smallLambdaHat(:,s);
                    obj.bigLambdaBar(:,:,s,n) = hinvAR(:,s)*h' + B(:,:,s)'*bigLambdaHat(:,:,s)*B(:,:,s);

                    %Values for the VB updates: calculate the alphas
                    %each class by setting inputs to 1 for each class
                    r_tvb(:,s) = diag(P_post(:,:,s,n));
%                     r_tN(s) = h' * P_post(:,:,s,n) * h;                    
                end
%                 k_tN = (1./r_tN) .* (1+exp(h'*Wmean_post(:,:,n)));
                k_tvb = (1./r_tvb) .* (1+exp(Wmean_post(:,:,n)));
                m_tvb = k_tvb .* (1+exp(-Wmean_post(:,:,n)));

                obj.Alpha(:, :, n) = k_tvb;
                obj.AlphaSum(:, :, n) = m_tvb;                  
            end       
            
            obj.C3 = obj.C{3};
        end
        
        function [pT, lnPT] = expectedT(obj)     
                        
            nonZeroScores = find(obj.C{3}~=0);
            nValid = length(nonZeroScores);
            
            validDecs = obj.C{3}(nonZeroScores); % c
            validN = obj.C{2}(nonZeroScores); % index of data points

            indx = sub2ind([obj.nScores length(nonZeroScores)], ...
                            validDecs, ...
                            nonZeroScores);     
            lnPT = zeros(obj.nClasses, obj.nObjects);
            
            for j=1:obj.nClasses                
                lnPT(j,:) = sparse(ones(1,nValid), validN', obj.lnPi(j, indx), 1, obj.nObjects);
                lnPT(j,validN) = lnPT(j,validN) + obj.lnKappa(j); %used to have realIdxs in place of validN
            end
            
            expA = exp(lnPT(:,validN));
            expB = repmat(sum(expA,1), obj.nClasses, 1);
            
            %stop any extreme values from causing Na
            expB(expA==Inf) = 1;
            expA(expA==Inf) = 1;
            
            pT = sparse(obj.nClasses, obj.nObjects);
            pT(:, validN) = expA./expB;
                                
            if length(obj.targets)<1
                obj.post_T = pT; 
            else
                obj.post_T = obj.Tmat;
                obj.post_T(:, obj.testIdxs) = pT(:, obj.testIdxs);
            end
        end          
            
        function [L, EEnergy, H] = lowerBound(obj)

            nResponses = length(obj.C{1});
            
            lnPi_t = zeros(size(obj.lnPi,2),obj.nObjects);
            for i=1:obj.nObjects
                lnPi_t(:,i) = sum( obj.lnPi(:,:,i) .* repmat(obj.post_T(:,i), 1, size(obj.lnPi,2)) )';
            end
            
            idxs = sub2ind([obj.nScores nResponses], obj.C{3}, (1:nResponses)');            
            ElnPC_t = lnPi_t(idxs);
            ElnPC = sum(ElnPC_t);
            
            if size(obj.lnKappa, 1) > size(obj.lnKappa, 2)
                obj.lnKappa = obj.lnKappa';
            end
            TargetCounts = sum(obj.post_T, 2);            
            ElnPT = sum(TargetCounts .* obj.lnKappa', 1);
        
            AlphaPr = obj.updateAlpha(true);
            ElnPPi_t = gammaln(sum(AlphaPr, 1))-sum(gammaln(AlphaPr),1) + sum((AlphaPr-1).*lnPi_t, 1);  
            ElnPPi = sum(ElnPPi_t);
            
            ElnPP = gammaln(sum(obj.Nu0, 2))-sum(gammaln(obj.Nu0),2) + sum((obj.Nu0-1).*obj.lnKappa, 2);
            
            EEnergy = ElnPC + ElnPT + ElnPPi + ElnPP;
        
            if size(obj.lnKappa, 1) > size(obj.lnKappa, 2)
                obj.lnKappa = obj.lnKappa';
            end            

            ElnQT = sum(sum(obj.post_T(obj.post_T~=0) .* log(obj.post_T(obj.post_T~=0))));
            
            AlphaPost_t = zeros(size(obj.lnPi,2),obj.nObjects);
            for i=1:obj.nObjects
                AlphaPost_t(:,i) = sum( obj.Alpha(:,:,i) .* repmat(obj.post_T(:,i), 1, size(obj.Alpha,2)) )';
            end                                   
            ElnQPi_t = gammaln(sum(AlphaPost_t, 1))-sum(gammaln(AlphaPost_t),1) ...
                + sum((AlphaPost_t-1).*lnPi_t, 1);
            ElnQPi = sum(ElnQPi_t);

            PostNu = obj.Nu0 + TargetCounts';
            ElnQP = gammaln(sum(PostNu, 2))-sum(gammaln(PostNu),2) + sum((PostNu-1).*obj.lnKappa, 2);
            
            H = - ElnQT - ElnQPi - ElnQP;
            
            L = EEnergy + H;
        end
        
        function printPiEvolution(obj)
                                
            classifiers = unique(obj.obj.C{1});
            for k=classifiers'
                display(['Pi for classifier ' num2str(k)]);
                kidxs = find(obj.C{1}==k)';
                for n=[kidxs(1) kidxs(round(length(kidxs)/2)) kidxs(end)]
                    display([num2str(n) ',  ' num2str(obj.C{3}(n)) ',  ' num2str(obj.targets(obj.C{2}(n)))]);
                    obj.Alpha(:,:,n)
                end
            end            
        end        
    end 
end

