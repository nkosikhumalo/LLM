package com.microllm.optim;

import com.microllm.tensor.Tensor;
import java.util.ArrayList;
import java.util.List;

/** Adam optimizer with decoupled weight decay. */
public final class AdamW implements Optimizer {
    private final List<Tensor> parameters;
    private final List<double[]> firstMoments = new ArrayList<>();
    private final List<double[]> secondMoments = new ArrayList<>();
    private double learningRate;
    private final double beta1, beta2, epsilon, weightDecay;
    private int step;

    public AdamW(List<Tensor> parameters, double learningRate) {
        this(parameters, learningRate, 0.9, 0.999, 1e-8, 0.01);
    }

    public AdamW(List<Tensor> parameters, double learningRate, double beta1, double beta2, double epsilon, double weightDecay) {
        if (parameters == null || learningRate <= 0.0 || !Double.isFinite(learningRate)
                || beta1 < 0.0 || beta1 >= 1.0 || beta2 < 0.0 || beta2 >= 1.0
                || epsilon <= 0.0 || weightDecay < 0.0) {
            throw new IllegalArgumentException("invalid AdamW parameters");
        }
        this.parameters = List.copyOf(parameters);
        this.learningRate = learningRate; this.beta1 = beta1; this.beta2 = beta2;
        this.epsilon = epsilon; this.weightDecay = weightDecay;
        for (Tensor parameter : this.parameters) {
            firstMoments.add(new double[parameter.size()]);
            secondMoments.add(new double[parameter.size()]);
        }
    }

    public void setLearningRate(double learningRate) {
        if (learningRate < 0.0 || !Double.isFinite(learningRate)) {
            throw new IllegalArgumentException("learning rate must be finite and nonnegative");
        }
        this.learningRate = learningRate;
    }

    @Override public void step() {
        step++;
        double firstCorrection = 1.0 - Math.pow(beta1, step);
        double secondCorrection = 1.0 - Math.pow(beta2, step);
        for (int p = 0; p < parameters.size(); p++) {
            Tensor parameter = parameters.get(p);
            double[] gradient = parameter.gradient();
            double[] first = firstMoments.get(p), second = secondMoments.get(p);
            for (int i = 0; i < parameter.size(); i++) {
                first[i] = beta1 * first[i] + (1.0 - beta1) * gradient[i];
                second[i] = beta2 * second[i] + (1.0 - beta2) * gradient[i] * gradient[i];
                double mean = first[i] / firstCorrection;
                double variance = second[i] / secondCorrection;
                double value = parameter.get(i);
                parameter.set(i, value - learningRate * (mean / (Math.sqrt(variance) + epsilon) + weightDecay * value));
            }
        }
    }

    @Override public void zeroGrad() { for (Tensor parameter : parameters) parameter.zeroGradient(); }
}
