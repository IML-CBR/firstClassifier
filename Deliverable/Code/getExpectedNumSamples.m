function num_samples = getExpectedNumSamples(dev_error,vc,confidence)
    gen_error = 1-confidence;
    curr_gen_error = Inf;
    
    num_samples = 50;
    while curr_gen_error > gen_error
        numerator = vc*(log(2*num_samples/vc)+1)+log(2/dev_error);
        denomiator = 2*num_samples;
        curr_gen_error = sqrt(numerator/denomiator);
        num_samples = num_samples + 50;
    end
end