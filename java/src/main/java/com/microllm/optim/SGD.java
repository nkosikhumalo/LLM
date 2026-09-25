package com.microllm.optim;

import com.microllm.tensor.Tensor;
import java.util.List;

/** Stochastic gradient descent over a fixed set of trainable tensors. */
public final class SGD implements Optimizer {
    private final List<Tensor> parameters;
    private final double learningRate;

    public SGD(List<Tensor> parameters, double learningRate) {
        if (learningRate <= 0.0 || !Double.isFinite(learningRate)) throw new IllegalArgumentException("learning rate must be finite and positive");
        this.parameters = List.copyOf(parameters);
        this.learningRate = learningRate;
    }

    @Override public void step() {
        for (Tensor parameter : parameters) {
            for (int i = 0; i < parameter.size(); i++) parameter.set(i, parameter.get(i) - learningRate * parameter.gradient()[i]);
        }
    }

    @Override public void zeroGrad() { for (Tensor parameter : parameters) parameter.zeroGradient(); }
}
