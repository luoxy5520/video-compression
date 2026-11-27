# Optimizer audit

Main/aux overlap:0; coverage:31/31 trainable tensors. Main Adam lr.001; auxiliary Adam lr.001. Quantiles excluded from main, exclusively auxiliary. GaussianConditional has no trainable parameters; hyper decoder predicts scales. No AMP used in historical codec training. Global gradient clip1 makes different high-lambda runs similar in scale and can obscure rate gradients; this is optimization behavior, not a swapped-loss bug.
