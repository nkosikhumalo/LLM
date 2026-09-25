package com.microllm.tensor;

import java.util.Arrays;
import java.util.Random;

/** Dense, row-major tensor with a gradient buffer for trainable parameters. */
public final class Tensor {
    private final int[] shape;
    private final double[] values;
    private final double[] gradient;

    public Tensor(int... shape) {
        this(new double[elementCount(shape)], shape);
    }

    public Tensor(double[] values, int... shape) {
        int expected = elementCount(shape);
        if (values.length != expected) {
            throw new IllegalArgumentException("value count does not match tensor shape");
        }
        this.shape = shape.clone();
        this.values = values.clone();
        this.gradient = new double[values.length];
    }

    public static Tensor zeros(int... shape) { return new Tensor(shape); }

    public static Tensor ones(int... shape) {
        Tensor tensor = new Tensor(shape);
        Arrays.fill(tensor.values, 1.0);
        return tensor;
    }

    public static Tensor randomNormal(Random random, double standardDeviation, int... shape) {
        if (random == null || standardDeviation < 0.0 || !Double.isFinite(standardDeviation)) {
            throw new IllegalArgumentException("random source and finite nonnegative deviation are required");
        }
        Tensor tensor = new Tensor(shape);
        for (int i = 0; i < tensor.values.length; i++) {
            tensor.values[i] = random.nextGaussian() * standardDeviation;
        }
        return tensor;
    }

    public int size() { return values.length; }
    public int rank() { return shape.length; }
    public int dimension(int axis) { return shape[axis]; }
    public int[] shape() { return shape.clone(); }
    public double get(int index) { return values[index]; }
    public void set(int index, double value) { values[index] = value; }
    public double[] values() { return values; }
    public double[] gradient() { return gradient; }
    public void zeroGradient() { Arrays.fill(gradient, 0.0); }

    private static int elementCount(int[] shape) {
        if (shape == null || shape.length == 0) {
            throw new IllegalArgumentException("tensor shape must have at least one dimension");
        }
        int count = 1;
        for (int dimension : shape) {
            if (dimension < 1) throw new IllegalArgumentException("tensor dimensions must be positive");
            count = Math.multiplyExact(count, dimension);
        }
        return count;
    }
}
