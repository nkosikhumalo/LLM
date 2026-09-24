package com.microllm.layers;

/**
 * Belongs here: Layer Normalization — keeps activations in a stable numeric range.
 *
 * Intended contents:
 * - Learnable scale (gamma) and bias (beta)
 * - Forward mean/variance normalize + affine
 * - Backward for training
 *
 * Alternative: RMSNorm (same package or sibling class) if you prefer modern LLM norms.
 *
 * Used by: model.Transformer (pre/post attention and FFN)
 */
public class LayerNorm {
}
