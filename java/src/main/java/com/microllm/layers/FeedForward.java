package com.microllm.layers;

/**
 * Belongs here: position-wise feed-forward network (FFN / MLP block).
 *
 * Intended contents:
 * - Expand linear → non-linearity (GELU or SwiGLU) → project back to d_model
 * - Forward + backward
 *
 * Used by: model.Transformer blocks (after attention)
 */
public class FeedForward {
}
