package com.microllm.autograd;

import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Set;

/** Reverse-mode backpropagation for graphs built from {@link Variable} operations. */
public final class AutogradEngine {
    private AutogradEngine() { }

    /** Backpropagates from a scalar variable, accumulating into its leaf tensors. */
    public static void backward(Variable output) {
        if (output.data().size() != 1) throw new IllegalArgumentException("backward requires a scalar output");
        List<Variable> order = new ArrayList<>();
        Set<Variable> visited = Collections.newSetFromMap(new IdentityHashMap<>());
        visit(output, visited, order);
        output.data().gradient()[0] += 1.0;
        for (int node = order.size() - 1; node >= 0; node--) {
            Variable current = order.get(node);
            double[] gradient = current.data().gradient();
            for (Variable.Edge edge : current.edges()) {
                double[] contribution = edge.rule().apply(gradient);
                double[] parentGradient = edge.parent().data().gradient();
                for (int i = 0; i < parentGradient.length; i++) parentGradient[i] += contribution[i];
            }
        }
    }

    private static void visit(Variable node, Set<Variable> visited, List<Variable> order) {
        if (!visited.add(node)) return;
        for (Variable.Edge edge : node.edges()) visit(edge.parent(), visited, order);
        order.add(node);
    }
}
