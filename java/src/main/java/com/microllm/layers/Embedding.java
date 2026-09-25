package com.microllm.layers;

import com.microllm.tensor.Tensor;

/** Token embedding table with lookup and gradient accumulation. */
public final class Embedding {
    private final Tensor weights;

    public Embedding(Tensor weights) {
        if (weights.rank() != 2) throw new IllegalArgumentException("embedding weights must be a matrix");
        this.weights = weights;
    }

    public double[][] forward(int[] tokenIds) {
        double[][] result = new double[tokenIds.length][weights.dimension(1)];
        for (int position = 0; position < tokenIds.length; position++) {
            int token = tokenIds[position];
            if (token < 0 || token >= weights.dimension(0)) throw new IllegalArgumentException("token ID outside vocabulary");
            System.arraycopy(weights.values(), token * weights.dimension(1), result[position], 0, weights.dimension(1));
        }
        return result;
    }

    public void backward(int[] tokenIds, double[][] gradient) {
        if (gradient.length != tokenIds.length) throw new IllegalArgumentException("gradient sequence length mismatch");
        int width = weights.dimension(1);
        for (int position = 0; position < tokenIds.length; position++) {
            if (gradient[position].length != width) throw new IllegalArgumentException("gradient width mismatch");
            int token = tokenIds[position];
            if (token < 0 || token >= weights.dimension(0)) throw new IllegalArgumentException("token ID outside vocabulary");
            for (int i = 0; i < width; i++) weights.gradient()[token * width + i] += gradient[position][i];
        }
    }

    public Tensor weights() { return weights; }
}
