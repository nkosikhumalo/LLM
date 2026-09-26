package com.microllm.export;

import com.microllm.model.ModelConfig;
import com.microllm.model.Transformer;
import com.microllm.tensor.Tensor;

import java.io.DataInputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/** Reads the compact, validated weight bridge written by the Go QAT command. */
public final class CheckpointImporter {
    private static final byte[] MAGIC = "MINIQAT1".getBytes(java.nio.charset.StandardCharsets.US_ASCII);

    private CheckpointImporter() { }

    public static LoadedCheckpoint read(Path path) throws IOException {
        try (DataInputStream input = new DataInputStream(Files.newInputStream(path))) {
            byte[] magic = input.readNBytes(MAGIC.length);
            if (!java.util.Arrays.equals(magic, MAGIC)) throw new IOException("invalid QAT weight bridge signature");
            ModelConfig config;
            try {
                config = new ModelConfig(input.readInt(), input.readInt(), input.readInt(), input.readInt(),
                        input.readInt(), input.readInt(), input.readDouble());
            } catch (IllegalArgumentException exception) {
                throw new IOException("invalid QAT model config", exception);
            }
            int tensorCount = input.readInt();
            if (tensorCount != 4 + config.nLayers() * 8) throw new IOException("QAT tensor count does not match config");
            Transformer shapeModel = new Transformer(config, 0L);
            List<Tensor> expected = shapeModel.trainableParameters();
            List<double[]> tensors = new ArrayList<>(tensorCount);
            for (int tensorIndex = 0; tensorIndex < tensorCount; tensorIndex++) {
                int length = input.readInt();
                if (length != expected.get(tensorIndex).size()) {
                    throw new IOException("QAT tensor " + tensorIndex + " has " + length + " values; expected " + expected.get(tensorIndex).size());
                }
                double[] values = new double[length];
                for (int valueIndex = 0; valueIndex < length; valueIndex++) {
                    values[valueIndex] = input.readDouble();
                    if (!Double.isFinite(values[valueIndex])) throw new IOException("QAT weights contain a non-finite value");
                }
                tensors.add(values);
            }
            if (input.read() != -1) throw new IOException("unexpected trailing bytes in QAT weight bridge");
            return new LoadedCheckpoint(config, List.copyOf(tensors));
        }
    }

    public record LoadedCheckpoint(ModelConfig config, List<double[]> tensors) { }
}
