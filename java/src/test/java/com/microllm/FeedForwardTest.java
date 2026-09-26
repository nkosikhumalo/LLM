package com.microllm;

import com.microllm.layers.FeedForward;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class FeedForwardTest {
    @Test
    void geluForwardMatchesKnownValue() {
        double[] in = {1, 0, 0, 1}; // identity 2x2
        double[] out = {1, 0, 0, 1};
        FeedForward ffn = new FeedForward(2, 2, in, out);
        FeedForward.Cache cache = ffn.forward(new double[][]{{1.0, 0.0}});
        assertEquals(1, cache.output().length);
        assertEquals(2, cache.output()[0].length);
        // GELU(1) ≈ 0.84119 through identity in/out
        assertEquals(0.84119199, cache.output()[0][0], 1e-5);
        assertEquals(0.0, cache.output()[0][1], 1e-12);
    }

    @Test
    void backwardFiniteDifferenceOnInWeight() {
        double[] inWeight = {0.2, -0.1, 0.05, 0.3};
        double[] outWeight = {0.4, -0.2, 0.1, 0.25};
        FeedForward ffn = new FeedForward(2, 2, inWeight, outWeight);
        double[][] input = {{0.5, -0.25}};
        FeedForward.Cache cache = ffn.forward(input);
        double[][] gradOut = {{1.0, -0.5}};
        FeedForward.Gradients grads = ffn.backward(cache, gradOut);

        double epsilon = 1e-5;
        double[] numerical = new double[inWeight.length];
        for (int i = 0; i < inWeight.length; i++) {
            double[] plus = inWeight.clone();
            double[] minus = inWeight.clone();
            plus[i] += epsilon;
            minus[i] -= epsilon;
            double lossPlus = scalarLoss(new FeedForward(2, 2, plus, outWeight).forward(input).output()[0], gradOut[0]);
            double lossMinus = scalarLoss(new FeedForward(2, 2, minus, outWeight).forward(input).output()[0], gradOut[0]);
            numerical[i] = (lossPlus - lossMinus) / (2 * epsilon);
        }
        assertArrayEquals(numerical, grads.inWeight(), 1e-4);
        assertTrue(Math.abs(grads.inWeight()[0]) + Math.abs(grads.outWeight()[0]) > 0);
    }

    private static double scalarLoss(double[] output, double[] gradOut) {
        double sum = 0.0;
        for (int i = 0; i < output.length; i++) {
            sum += output[i] * gradOut[i];
        }
        return sum;
    }
}
