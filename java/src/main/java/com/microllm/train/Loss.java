package com.microllm.train;

import com.microllm.tensor.Ops;

/** Cross-entropy for next-token prediction with a logits gradient. */
public final class Loss {
    private Loss() { }

    public static Result crossEntropy(double[][] logits, int[] targets) {
        if (logits.length == 0 || logits.length != targets.length) throw new IllegalArgumentException("logits and targets must have the same nonzero batch size");
        double[][] gradient = new double[logits.length][];
        double loss = 0.0;
        for (int row = 0; row < logits.length; row++) {
            if (targets[row] < 0 || targets[row] >= logits[row].length) throw new IllegalArgumentException("target outside vocabulary");
            double[] probabilities = Ops.softmax(logits[row]);
            loss -= Math.log(Math.max(probabilities[targets[row]], 1e-300));
            gradient[row] = probabilities;
            gradient[row][targets[row]] -= 1.0;
            for (int i = 0; i < gradient[row].length; i++) gradient[row][i] /= logits.length;
        }
        return new Result(loss / logits.length, gradient);
    }

    public record Result(double value, double[][] gradient) { }
}
