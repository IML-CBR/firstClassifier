function error_bound = errorBound(error_train,Dvc,num_samples,error_dev)
    VC_part = Dvc*(log(2*num_samples/Dvc)+1);
    error_bound = (VC_part+log(2/error_dev))/(2*num_samples);
    error_bound = error_bound + error_train;
end