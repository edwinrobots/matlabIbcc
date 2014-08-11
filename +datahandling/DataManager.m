classdef DataManager
    %DATAMANAGER Takes a bunch of raw data and converts it into a simplified
    %format for use by IBCC etc., and converts back. Translates the IDs
    %into consecutive numbers etc.
    
    properties
        origIds = [] %set of unique IDs
        rawIds = [] %the list of IDs from the dataset
        localIdxs = [] %mapped list of local indexes for the dataset
    end
    
    methods
        
        function obj=DataManager(rawData)
            %raw data is a list of values from a dataset containing the IDs
            %of objects that each data point relates to.
            
            %Get the complete set of unique IDs.
            [obj.origIds, ~, obj.localIdxs] = unique(rawData);
            
            obj.rawIds = rawData;
        end
        
        function ids=mapAllToOrigIds(obj)
            ids = obj.origIds;
        end
        
        function ids=getOrigIdsFromIdxs(obj, idxsToSearch)
            ids = obj.origIds(idxsToSearch);
        end
        
        function idxs=getIdxsForOrigIds(obj, idsToSearch)
           %get the local indexes from the original IDs
           
           idxs = zeros(length(idsToSearch),1);
           
           for i=1:length(idsToSearch)
               idxs(i) = find(obj.origIds==idstoSearch(i));
           end
        end
        
        
        
    end
    
end

