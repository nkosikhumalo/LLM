package com.microllm.autograd;

/**
 * Belongs here: a value in the computational graph (tensor + grad + how it was produced).
 *
 * Intended contents:
 * - Reference to data Tensor and grad Tensor
 * - Optional parent ops / Function for backward
 * - requiresGrad flag
 *
 * Used by: AutogradEngine and layer forwards that participate in training
 */
public class Variable {
}
