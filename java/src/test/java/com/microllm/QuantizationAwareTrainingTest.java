package com.microllm;

import com.microllm.model.ModelConfig;
import com.microllm.model.Transformer;
import com.microllm.optim.AdamW;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class QuantizationAwareTrainingTest {
    @Test
    void fakeQuantizesWeightsToSignedInt8Grid() {
        double[] original = {-0.91, -0.22, 0.13, 0.89};
        double[] quantized = Transformer.fakeQuantizeWeights(original);
        assertNotEquals(Arrays.toString(original), Arrays.toString(quantized));
        double scale = 0.91 / 127.0;
        for (double value : quantized) {
            double units = value / scale;
            assertTrue(units >= -127.0000001 && units <= 127.0000001);
            assertTrue(Math.abs(units - Math.rint(units)) < 1e-8);
        }
    }

    @Test
    void fakeQuantizesMatrixRowsWithIndependentScales() {
        double[] values = {0.001, -0.002, 1.0, -0.5};
        double[] quantized = Transformer.fakeQuantizeWeights(values, 2, 2);
        double firstRowScale = 0.002 / 127.0;
        double secondRowScale = 1.0 / 127.0;
        assertTrue(Math.abs(values[0] - quantized[0]) <= firstRowScale / 2.0 + 1e-12);
        assertTrue(Math.abs(values[1] - quantized[1]) <= firstRowScale / 2.0 + 1e-12);
        assertTrue(Math.abs(values[2] - quantized[2]) <= secondRowScale / 2.0 + 1e-12);
        assertTrue(Math.abs(values[3] - quantized[3]) <= secondRowScale / 2.0 + 1e-12);
    }

    @Test
    void fakeQuantizedForwardStillBackpropagatesToLatentWeights() {
        Transformer model = new Transformer(new ModelConfig(8, 4, 1, 2, 8, 8, 1e-5), 123);
        model.setFakeQuantization(true);
        double[] before = model.output().values().clone();
        for (var parameter : model.trainableParameters()) parameter.zeroGradient();

        double loss = model.trainWindow(new int[]{0, 1, 2, 3}, new int[]{1, 2, 3, 4}, 0);
        new AdamW(model.trainableParameters(), 1e-3).step();

        assertTrue(Double.isFinite(loss));
        boolean changed = false;
        for (int i = 0; i < before.length; i++) changed |= before[i] != model.output().values()[i];
        assertTrue(changed, "straight-through gradients should update latent FP32 weights");
        assertFalse(Arrays.equals(before, model.output().values()));
    }
}
