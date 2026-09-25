package com.microllm.tensor;

/** Numerically stable tensor operations shared by the model and loss. */
public final class Ops {
    private Ops() { }

    public static double[] softmax(double[] logits) {
        if (logits == null || logits.length == 0) {
            throw new IllegalArgumentException("softmax requires at least one logit");
        }
        double max = Double.NEGATIVE_INFINITY;
        for (double value : logits) {
            if (!Double.isFinite(value)) throw new IllegalArgumentException("logits must be finite");
            max = Math.max(max, value);
        }
        double[] probabilities = new double[logits.length];
        double sum = 0.0;
        for (int i = 0; i < logits.length; i++) {
            probabilities[i] = Math.exp(logits[i] - max);
            sum += probabilities[i];
        }
        for (int i = 0; i < probabilities.length; i++) probabilities[i] /= sum;
        return probabilities;
    }
}
