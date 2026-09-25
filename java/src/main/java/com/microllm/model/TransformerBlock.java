package com.microllm.model;

/** Defines the flat per-layer weight layout shared by model training and export. */
public final class TransformerBlock {
    public static final int WEIGHT_COUNT = 8;
    public static final int ATTENTION_NORM = 0;
    public static final int QUERY = 1;
    public static final int KEY = 2;
    public static final int VALUE = 3;
    public static final int ATTENTION_OUTPUT = 4;
    public static final int FEED_FORWARD_NORM = 5;
    public static final int FEED_FORWARD_INPUT = 6;
    public static final int FEED_FORWARD_OUTPUT = 7;

    private TransformerBlock() { }

    public static int weightOffset(int layerIndex) {
        if (layerIndex < 0) throw new IllegalArgumentException("layer index must be nonnegative");
        return Math.multiplyExact(layerIndex, WEIGHT_COUNT);
    }
}
