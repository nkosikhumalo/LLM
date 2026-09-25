package com.microllm.layers;

import com.microllm.tensor.Tensor;

/** RMS normalization with learnable per-feature scale and analytic backward pass. */
public final class RMSNorm {
    private final Tensor scale;
    private final double epsilon;

    public RMSNorm(Tensor scale, double epsilon) {
        if (scale.rank() != 1 || epsilon <= 0.0 || !Double.isFinite(epsilon)) throw new IllegalArgumentException("invalid RMSNorm parameters");
        this.scale = scale;
        this.epsilon = epsilon;
    }

    public Cache forward(double[] input) {
        checkWidth(input);
        double meanSquare = 0.0;
        for (double value : input) meanSquare += value * value;
        meanSquare /= input.length;
        double inverseRms = 1.0 / Math.sqrt(meanSquare + epsilon);
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++) output[i] = input[i] * inverseRms * scale.get(i);
        return new Cache(input.clone(), inverseRms, output);
    }

    public double[] backward(Cache cache, double[] outputGradient) {
        checkWidth(outputGradient);
        double dot = 0.0;
        for (int i = 0; i < outputGradient.length; i++) {
            dot += outputGradient[i] * scale.get(i) * cache.input()[i];
            scale.gradient()[i] += outputGradient[i] * cache.input()[i] * cache.inverseRms();
        }
        double[] inputGradient = new double[outputGradient.length];
        double correction = dot * cache.inverseRms() * cache.inverseRms() * cache.inverseRms() / outputGradient.length;
        for (int i = 0; i < inputGradient.length; i++) inputGradient[i] = outputGradient[i] * scale.get(i) * cache.inverseRms() - cache.input()[i] * correction;
        return inputGradient;
    }

    private void checkWidth(double[] values) { if (values.length != scale.size()) throw new IllegalArgumentException("feature width mismatch"); }
    public record Cache(double[] input, double inverseRms, double[] output) { }
}
