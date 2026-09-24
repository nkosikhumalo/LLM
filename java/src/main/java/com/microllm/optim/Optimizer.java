package com.microllm.optim;
/** Updates parameters after their gradient buffers have been accumulated. */
public interface Optimizer { void step(); void zeroGrad(); }
