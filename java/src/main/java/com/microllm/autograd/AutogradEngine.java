package com.microllm.autograd;

/**
 * Belongs here: backpropagation / autograd engine.
 *
 * Intended contents:
 * - Record computational ops during the forward pass (tape or graph nodes)
 * - Backward: walk the tape and apply the chain rule to fill gradients
 * - Helpers to attach Variable wrappers to Tensor that require grad
 *
 * Used by: train loop after loss is computed
 */
public class AutogradEngine {
}
