package com.microllm.layers;

import com.microllm.tensor.Ops;

/** Stable softmax over a single vector of logits. */
public final class Softmax {
    public double[] forward(double[] logits) { return Ops.softmax(logits); }
}
