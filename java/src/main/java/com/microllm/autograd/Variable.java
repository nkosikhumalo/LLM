package com.microllm.autograd;

import com.microllm.tensor.Tensor;
import java.util.ArrayList;
import java.util.List;

/** Tensor value and the local gradient rules linking it to its inputs. */
public final class Variable {
    @FunctionalInterface interface GradientRule { double[] apply(double[] outputGradient); }
    record Edge(Variable parent, GradientRule rule) { }

    private final Tensor data;
    private final boolean requiresGrad;
    private final List<Edge> edges = new ArrayList<>();

    public Variable(Tensor data, boolean requiresGrad) { this.data = data; this.requiresGrad = requiresGrad; }
    public Tensor data() { return data; }
    public boolean requiresGrad() { return requiresGrad; }

    public Variable add(Variable other) {
        requireSameShape(other);
        Tensor result = new Tensor(data.shape());
        for (int i = 0; i < data.size(); i++) result.set(i, data.get(i) + other.data.get(i));
        Variable output = new Variable(result, requiresGrad || other.requiresGrad);
        if (requiresGrad) output.edges.add(new Edge(this, gradient -> gradient.clone()));
        if (other.requiresGrad) output.edges.add(new Edge(other, gradient -> gradient.clone()));
        return output;
    }

    public Variable multiply(Variable other) {
        requireSameShape(other);
        Tensor result = new Tensor(data.shape());
        for (int i = 0; i < data.size(); i++) result.set(i, data.get(i) * other.data.get(i));
        Variable output = new Variable(result, requiresGrad || other.requiresGrad);
        if (requiresGrad) output.edges.add(new Edge(this, gradient -> multiply(gradient, other.data.values())));
        if (other.requiresGrad) output.edges.add(new Edge(other, gradient -> multiply(gradient, data.values())));
        return output;
    }

    public Variable sum() {
        Tensor result = new Tensor(1);
        double total = 0.0; for (double value : data.values()) total += value;
        result.set(0, total);
        Variable output = new Variable(result, requiresGrad);
        if (requiresGrad) output.edges.add(new Edge(this, gradient -> {
            double[] expanded = new double[data.size()];
            java.util.Arrays.fill(expanded, gradient[0]);
            return expanded;
        }));
        return output;
    }

    List<Edge> edges() { return edges; }
    private void requireSameShape(Variable other) {
        if (!java.util.Arrays.equals(data.shape(), other.data.shape())) throw new IllegalArgumentException("autograd operands must have matching shapes");
    }
    private static double[] multiply(double[] left, double[] right) {
        double[] result = new double[left.length];
        for (int i = 0; i < result.length; i++) result[i] = left[i] * right[i];
        return result;
    }
}
