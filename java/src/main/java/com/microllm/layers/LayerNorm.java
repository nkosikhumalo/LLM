package com.microllm.layers;

import com.microllm.tensor.Tensor;

/** Layer normalization with learnable scale and bias. */
public final class LayerNorm {
    private final Tensor scale;
    private final Tensor bias;
    private final double epsilon;

    public LayerNorm(Tensor scale, Tensor bias, double epsilon) {
        if (scale.rank() != 1 || bias.rank() != 1 || scale.size() != bias.size() || epsilon <= 0.0) {
            throw new IllegalArgumentException("invalid LayerNorm parameters");
        }
        this.scale = scale; this.bias = bias; this.epsilon = epsilon;
    }

    public Cache forward(double[] input) {
        checkWidth(input);
        double mean = 0.0; for (double value : input) mean += value; mean /= input.length;
        double variance = 0.0; for (double value : input) { double delta = value - mean; variance += delta * delta; }
        variance /= input.length;
        double inverseStd = 1.0 / Math.sqrt(variance + epsilon);
        double[] normalized = new double[input.length], output = new double[input.length];
        for (int i = 0; i < input.length; i++) { normalized[i] = (input[i] - mean) * inverseStd; output[i] = normalized[i] * scale.get(i) + bias.get(i); }
        return new Cache(normalized, inverseStd, output);
    }

    public double[] backward(Cache cache, double[] gradient) {
        checkWidth(gradient);
        int n = gradient.length; double sum = 0.0, normalizedDot = 0.0; double[] scaled = new double[n];
        for (int i = 0; i < n; i++) {
            scale.gradient()[i] += gradient[i] * cache.normalized()[i]; bias.gradient()[i] += gradient[i];
            scaled[i] = gradient[i] * scale.get(i); sum += scaled[i]; normalizedDot += scaled[i] * cache.normalized()[i];
        }
        double[] inputGradient = new double[n];
        for (int i = 0; i < n; i++) inputGradient[i] = cache.inverseStd() * (scaled[i] - sum / n - cache.normalized()[i] * normalizedDot / n);
        return inputGradient;
    }

    private void checkWidth(double[] input) { if (input.length != scale.size()) throw new IllegalArgumentException("feature width mismatch"); }
    public record Cache(double[] normalized, double inverseStd, double[] output) { }
}
