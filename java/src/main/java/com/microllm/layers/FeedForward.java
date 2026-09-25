package com.microllm.layers;

import java.util.Random;

/**
 * Position-wise GELU feed-forward network.
 * Matrices are row-major [inputDimension, outputDimension], matching Go inference.
 */
public final class FeedForward {
    private final int dModel;
    private final int dFfn;
    private final double[] inWeight;
    private final double[] outWeight;

    public FeedForward(int dModel, int dFfn, double[] inWeight, double[] outWeight) {
        if (inWeight.length != dModel * dFfn || outWeight.length != dFfn * dModel) {
            throw new IllegalArgumentException("FFN weight shapes do not match dModel/dFfn");
        }
        this.dModel = dModel;
        this.dFfn = dFfn;
        this.inWeight = inWeight;
        this.outWeight = outWeight;
    }

    public Cache forward(double[][] input) {
        return forward(input, 0.0, new Random(0));
    }

    public Cache forward(double[][] input, double dropoutRate, Random random) {
        if (dropoutRate < 0.0 || dropoutRate >= 1.0) throw new IllegalArgumentException("dropout rate must be in [0, 1)");
        int n = input.length;
        double[][] hidden = new double[n][];
        double[][] activated = new double[n][];
        double[][] dropoutMasks = new double[n][];
        double[][] output = new double[n][];
        for (int p = 0; p < n; p++) {
            hidden[p] = rowMat(input[p], inWeight, dModel, dFfn);
            activated[p] = new double[dFfn];
            dropoutMasks[p] = new double[dFfn];
            for (int i = 0; i < dFfn; i++) {
                double mask = dropoutRate == 0.0 || random.nextDouble() >= dropoutRate
                        ? 1.0 / (1.0 - dropoutRate) : 0.0;
                dropoutMasks[p][i] = mask;
                activated[p][i] = gelu(hidden[p][i]) * mask;
            }
            output[p] = rowMat(activated[p], outWeight, dFfn, dModel);
        }
        return new Cache(input, hidden, activated, dropoutMasks, output);
    }

    public Gradients backward(Cache cache, double[][] gradOutput) {
        int n = cache.input.length;
        double[][] gInput = new double[n][dModel];
        double[] gIn = new double[dModel * dFfn];
        double[] gOut = new double[dFfn * dModel];
        for (int p = 0; p < n; p++) {
            double[] gActivated = new double[dFfn];
            for (int i = 0; i < dFfn; i++) {
                for (int j = 0; j < dModel; j++) {
                    gOut[i * dModel + j] += cache.activated[p][i] * gradOutput[p][j];
                    gActivated[i] += outWeight[i * dModel + j] * gradOutput[p][j];
                }
            }
            double[] gHidden = new double[dFfn];
            for (int i = 0; i < dFfn; i++) {
                gHidden[i] = gActivated[i] * cache.dropoutMasks[p][i] * geluGrad(cache.hidden[p][i]);
            }
            for (int i = 0; i < dModel; i++) {
                for (int j = 0; j < dFfn; j++) {
                    gIn[i * dFfn + j] += cache.input[p][i] * gHidden[j];
                    gInput[p][i] += inWeight[i * dFfn + j] * gHidden[j];
                }
            }
        }
        return new Gradients(gInput, gIn, gOut);
    }

    private static double gelu(double x) {
        return 0.5 * x * (1.0 + Math.tanh(Math.sqrt(2.0 / Math.PI) * (x + 0.044715 * x * x * x)));
    }

    private static double geluGrad(double x) {
        double inner = Math.sqrt(2.0 / Math.PI) * (x + 0.044715 * x * x * x);
        double tanh = Math.tanh(inner);
        double sech2 = 1.0 - tanh * tanh;
        double dInner = Math.sqrt(2.0 / Math.PI) * (1.0 + 3.0 * 0.044715 * x * x);
        return 0.5 * (1.0 + tanh) + 0.5 * x * sech2 * dInner;
    }

    private static double[] rowMat(double[] x, double[] w, int rows, int cols) {
        double[] y = new double[cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                y[j] += x[i] * w[i * cols + j];
            }
        }
        return y;
    }

    public record Cache(double[][] input, double[][] hidden, double[][] activated, double[][] dropoutMasks, double[][] output) {}

    public record Gradients(double[][] input, double[] inWeight, double[] outWeight) {}
}
