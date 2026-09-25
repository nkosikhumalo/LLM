package com.microllm.layers;

import java.util.Random;

/** Causal multi-head attention and gradients for its four projections.
 * Projection matrices use row-major [inputDimension, outputDimension] layout. */
public final class MultiHeadAttention {
    private final int width;
    private final int heads;
    private final int headWidth;
    private final double[] qWeight, kWeight, vWeight, oWeight;

    public MultiHeadAttention(int width, int heads, double[] q, double[] k, double[] v, double[] o) {
        if (width < 1 || heads < 1 || width % heads != 0) throw new IllegalArgumentException("width must be positive and divisible by heads");
        int matrixSize = Math.multiplyExact(width, width);
        if (q.length != matrixSize || k.length != matrixSize || v.length != matrixSize || o.length != matrixSize) {
            throw new IllegalArgumentException("attention projection must be width by width");
        }
        this.width = width; this.heads = heads; this.headWidth = width / heads;
        this.qWeight = q; this.kWeight = k; this.vWeight = v; this.oWeight = o;
    }

    public Cache forward(double[][] input) {
        return forward(input, 0.0, new Random(0));
    }

    /** Dropout is applied to post-softmax attention probabilities during training only. */
    public Cache forward(double[][] input, double dropoutRate, Random random) {
        if (dropoutRate < 0.0 || dropoutRate >= 1.0) throw new IllegalArgumentException("dropout rate must be in [0, 1)");
        validateRows(input);
        int length = input.length;
        double[][] q = new double[length][], k = new double[length][], v = new double[length][];
        double[][] joined = new double[length][], output = new double[length][];
        double[][][] probabilities = new double[length][heads][];
        double[][][] usedProbabilities = new double[length][heads][];
        double[][][] dropoutMasks = new double[length][heads][];
        for (int position = 0; position < length; position++) {
            q[position] = rowMat(input[position], qWeight, width, width);
            k[position] = rowMat(input[position], kWeight, width, width);
            v[position] = rowMat(input[position], vWeight, width, width);
            joined[position] = new double[width];
            for (int head = 0; head < heads; head++) {
                double[] scores = new double[position + 1];
                for (int previous = 0; previous <= position; previous++) {
                    for (int dimension = 0; dimension < headWidth; dimension++) {
                        int index = head * headWidth + dimension;
                        scores[previous] += q[position][index] * k[previous][index];
                    }
                    scores[previous] /= Math.sqrt(headWidth);
                }
                softmaxInPlace(scores);
                probabilities[position][head] = scores;
                usedProbabilities[position][head] = scores.clone();
                dropoutMasks[position][head] = new double[scores.length];
                for (int previous = 0; previous <= position; previous++) {
                    double mask = dropoutRate == 0.0 || random.nextDouble() >= dropoutRate
                            ? 1.0 / (1.0 - dropoutRate) : 0.0;
                    dropoutMasks[position][head][previous] = mask;
                    usedProbabilities[position][head][previous] *= mask;
                    for (int dimension = 0; dimension < headWidth; dimension++) {
                        int index = head * headWidth + dimension;
                        joined[position][index] += usedProbabilities[position][head][previous] * v[previous][index];
                    }
                }
            }
            output[position] = rowMat(joined[position], oWeight, width, width);
        }
        return new Cache(input, q, k, v, joined, probabilities, usedProbabilities, dropoutMasks, output);
    }

    public Gradients backward(Cache cache, double[][] gradOutput) {
        int length = cache.input().length;
        validateRows(gradOutput);
        if (gradOutput.length != length) throw new IllegalArgumentException("output gradient sequence length mismatch");
        double[][] gInput = new double[length][width], gQ = new double[length][width];
        double[][] gK = new double[length][width], gV = new double[length][width];
        double[] gq = new double[width * width], gk = new double[width * width];
        double[] gv = new double[width * width], go = new double[width * width];

        for (int position = 0; position < length; position++) {
            double[] gJoined = new double[width];
            for (int i = 0; i < width; i++) {
                for (int j = 0; j < width; j++) {
                    go[i * width + j] += cache.joined()[position][i] * gradOutput[position][j];
                    gJoined[i] += oWeight[i * width + j] * gradOutput[position][j];
                }
            }
            for (int head = 0; head < heads; head++) {
                double[] gAttention = new double[position + 1];
                for (int previous = 0; previous <= position; previous++) {
                    for (int dimension = 0; dimension < headWidth; dimension++) {
                        int index = head * headWidth + dimension;
                        gAttention[previous] += gJoined[index] * cache.v()[previous][index];
                        gV[previous][index] += cache.usedProbabilities()[position][head][previous] * gJoined[index];
                    }
                    gAttention[previous] *= cache.dropoutMasks()[position][head][previous];
                }
                double weightedMean = 0.0;
                for (int previous = 0; previous <= position; previous++) {
                    weightedMean += gAttention[previous] * cache.probabilities()[position][head][previous];
                }
                for (int previous = 0; previous <= position; previous++) {
                    double scoreGradient = cache.probabilities()[position][head][previous]
                            * (gAttention[previous] - weightedMean) / Math.sqrt(headWidth);
                    for (int dimension = 0; dimension < headWidth; dimension++) {
                        int index = head * headWidth + dimension;
                        gQ[position][index] += scoreGradient * cache.k()[previous][index];
                        gK[previous][index] += scoreGradient * cache.q()[position][index];
                    }
                }
            }
        }

        for (int position = 0; position < length; position++) {
            for (int i = 0; i < width; i++) {
                for (int j = 0; j < width; j++) {
                    gq[i * width + j] += cache.input()[position][i] * gQ[position][j];
                    gk[i * width + j] += cache.input()[position][i] * gK[position][j];
                    gv[i * width + j] += cache.input()[position][i] * gV[position][j];
                    gInput[position][i] += qWeight[i * width + j] * gQ[position][j]
                            + kWeight[i * width + j] * gK[position][j]
                            + vWeight[i * width + j] * gV[position][j];
                }
            }
        }
        return new Gradients(gInput, gq, gk, gv, go);
    }

    private void validateRows(double[][] rows) {
        for (double[] row : rows) if (row.length != width) throw new IllegalArgumentException("attention input width mismatch");
    }

    private static double[] rowMat(double[] input, double[] weight, int rows, int columns) {
        double[] output = new double[columns];
        for (int i = 0; i < rows; i++) for (int j = 0; j < columns; j++) output[j] += input[i] * weight[i * columns + j];
        return output;
    }

    private static void softmaxInPlace(double[] values) {
        double max = Double.NEGATIVE_INFINITY;
        for (double value : values) max = Math.max(max, value);
        double sum = 0.0;
        for (int i = 0; i < values.length; i++) { values[i] = Math.exp(values[i] - max); sum += values[i]; }
        for (int i = 0; i < values.length; i++) values[i] /= sum;
    }

    public record Cache(double[][] input, double[][] q, double[][] k, double[][] v,
                        double[][] joined, double[][][] probabilities, double[][][] usedProbabilities,
                        double[][][] dropoutMasks, double[][] output) { }
    public record Gradients(double[][] input, double[] q, double[] k, double[] v, double[] o) { }
}
