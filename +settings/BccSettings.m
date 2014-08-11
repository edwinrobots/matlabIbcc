classdef BccSettings < handle
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        debug = false;
        
        %INFERENCE/MODELLING ----------------------------------------------       
        
        %HYPERPARAMETERS --------------------------------------------------
        Nu0 = [3 3];
        Alpha0 = [0.5 0.3 0.05; 0.18 0.36 0.41]; %*used in paper*

        %SCORE MAPPING ----------------------------------------------------
        IbccMapFunction = [];%@mapScoresDoNothing;         
        scoreMap = [];%[1 2 0; 0 1 -1];
        minScore = 0;
        maxScore = 1;         
        
        %EXPERT LABEL HANDLING --------------------------------------------
        %Make the last agents "trusted", i.e. expert labels with very low
        %probability of error
        trustedAgents = [];%indexes of trusted agents
        trustedAlpha = [];%specify prior hyperparams for trusted agent here
        
        %ITERATIONS -------------------------------------------------------
        convThreshold = 10^-2;
        maxIt = 10000;
        fixedNIt = 0; %250; %use a fixed number of iterations
        convIt = 2;
        
        %DYNAMIC IBCC -----------------------------------------------------
        changeRateMod = 1;
    end    
end

