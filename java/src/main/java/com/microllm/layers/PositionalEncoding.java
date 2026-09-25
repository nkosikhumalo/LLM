package com.microllm.layers;

import com.microllm.tensor.Tensor;

/** Adds learned absolute position vectors to token vectors. */
public final class PositionalEncoding {
    private final Tensor embeddings;

    public PositionalEncoding(Tensor embeddings) {
        if (embeddings.rank() != 2) throw new IllegalArgumentException("position embeddings must be a matrix");
        this.embeddings = embeddings;
    }

    public double[][] forward(double[][] input, int startPosition) {
        int width = embeddings.dimension(1);
        if (startPosition < 0 || startPosition + input.length > embeddings.dimension(0)) {
            throw new IllegalArgumentException("sequence exceeds position embedding range");
        }
        double[][] output = new double[input.length][width];
        for (int p = 0; p < input.length; p++) {
            if (input[p].length != width) throw new IllegalArgumentException("input width mismatch");
            for (int i = 0; i < width; i++) output[p][i] = input[p][i] + embeddings.get((startPosition + p) * width + i);
        }
        return output;
    }

    public void backward(double[][] gradient, int startPosition) {
        int width = embeddings.dimension(1);
        if (startPosition < 0 || startPosition + gradient.length > embeddings.dimension(0)) {
            throw new IllegalArgumentException("sequence exceeds position embedding range");
        }
        for (int p = 0; p < gradient.length; p++) {
            if (gradient[p].length != width) throw new IllegalArgumentException("gradient width mismatch");
            for (int i = 0; i < width; i++) embeddings.gradient()[(startPosition + p) * width + i] += gradient[p][i];
        }
    }

    public Tensor embeddings() { return embeddings; }
}
