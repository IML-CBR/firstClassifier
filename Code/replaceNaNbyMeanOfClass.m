function [ X_parsed ] = replaceNaNbyMeanOfClass( X, Y )
	% Replace the NaN values by the mean of the values of the other
	% instances of the same class
    X_parsed = X;
    classes = unique(Y);
    for i=1:size(X,1)
        prov_attribute = X(i,:);
<<<<<<< HEAD
        for j=1:size(classes,1)
=======
        for j=1:1:size(classes,1)
>>>>>>> 84e85dbcf5c7e386b75f33e1b253b81b224cec93
            instancesfromClass = find(Y==classes(j));
            if isnumeric(prov_attribute)
                notNaN = (intersect(instancesfromClass,find(~isnan(prov_attribute))))';
                yesNaN = (intersect(instancesfromClass,find(isnan(prov_attribute))))';
                if isempty(notNaN)
                    error('IN:replaceNaNbyMeanOfClassTrain',...
                    ['Error. \nAll values of one of the attributes are NaN.']);
                else
                    colMean = mean(prov_attribute(notNaN));
                    prov_attribute(yesNaN)=colMean;
                end
            else
                error('IN:replaceNaNbyMeanOfClassTrain',...
                    ['Error. \nThere are non numeric values in the'...
                    ' evaluated dataset.']);
            end
        end
        X_parsed(i,:) = prov_attribute;
    end
end

