package com.microllm.export;

import com.microllm.model.ModelConfig;
import com.microllm.model.Transformer;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;
import java.util.Random;

/**
 * Writes a versioned model.json which Go's internal/loader validates and reads.
 * Matrices are flattened row-major: input dimension first, output dimension second.
 */
public final class WeightExporter {
    private static final long DEFAULT_SEED = 42L;

    private WeightExporter() { }

    /** Writes deterministic, randomly initialized weights for pipeline verification. */
    public static void writeInitializedModel(Path outputPath, ModelConfig config) throws IOException {
        Random random = new Random(DEFAULT_SEED);
        StringBuilder json = new StringBuilder(64_000);
        json.append("{\n  \"format\": \"microllm\",\n  \"version\": 1,\n  \"config\": {")
                .append("\n    \"vocab_size\": ").append(config.vocabSize()).append(',')
                .append("\n    \"d_model\": ").append(config.dModel()).append(',')
                .append("\n    \"n_layers\": ").append(config.nLayers()).append(',')
                .append("\n    \"n_heads\": ").append(config.nHeads()).append(',')
                .append("\n    \"d_ffn\": ").append(config.dFfn()).append(',')
                .append("\n    \"max_seq_len\": ").append(config.maxSeqLen()).append(',')
                .append("\n    \"rms_norm_epsilon\": ").append(number(config.rmsNormEpsilon()))
                .append("\n  },\n  \"weights\": {");
        appendArray(json, "token_embedding", randomNormalArray(random, config.vocabSize() * config.dModel(), 0.02), true);
        appendArray(json, "position_embedding", randomNormalArray(random, config.maxSeqLen() * config.dModel(), 0.02), true);
        json.append("\n    \"layers\": [");
        for (int layerIndex = 0; layerIndex < config.nLayers(); layerIndex++) {
            if (layerIndex > 0) json.append(',');
            json.append("\n      {");
            appendArray(json, "attn_norm", ones(config.dModel()), true, 8);
            appendArray(json, "q", randomNormalArray(random, config.dModel() * config.dModel(), 0.02), true, 8);
            appendArray(json, "k", randomNormalArray(random, config.dModel() * config.dModel(), 0.02), true, 8);
            appendArray(json, "v", randomNormalArray(random, config.dModel() * config.dModel(), 0.02), true, 8);
            appendArray(json, "o", randomNormalArray(random, config.dModel() * config.dModel(), 0.02), true, 8);
            appendArray(json, "ffn_norm", ones(config.dModel()), true, 8);
            appendArray(json, "ffn_in", randomNormalArray(random, config.dModel() * config.dFfn(), 0.02), true, 8);
            appendArray(json, "ffn_out", randomNormalArray(random, config.dFfn() * config.dModel(), 0.02), false, 8);
            json.append("\n      }");
        }
        json.append("\n    ],");
        appendArray(json, "final_norm", ones(config.dModel()), true);
        appendArray(json, "output", randomNormalArray(random, config.dModel() * config.vocabSize(), 0.02), false);
        json.append("\n  }\n}\n");
        Files.createDirectories(outputPath.toAbsolutePath().getParent());
        Files.writeString(outputPath, json, StandardCharsets.UTF_8);
    }

    public static void writeModel(Path outputPath, Transformer model) throws IOException {
        ModelConfig config = model.config(); StringBuilder json = new StringBuilder(64000);
        json.append("{\n  \"format\": \"microllm\",\n  \"version\": 1,\n  \"config\": {").append("\n    \"vocab_size\": ").append(config.vocabSize()).append(",\n    \"d_model\": ").append(config.dModel()).append(",\n    \"n_layers\": ").append(config.nLayers()).append(",\n    \"n_heads\": ").append(config.nHeads()).append(",\n    \"d_ffn\": ").append(config.dFfn()).append(",\n    \"max_seq_len\": ").append(config.maxSeqLen()).append(",\n    \"rms_norm_epsilon\": ").append(number(config.rmsNormEpsilon())).append("\n  },\n  \"weights\": {");
        appendArray(json,"token_embedding",model.tokenEmbedding().values(),true); appendArray(json,"position_embedding",model.positionEmbedding().values(),true); json.append("\n    \"layers\": [");
        for(int l=0;l<config.nLayers();l++){int b=l*8;if(l>0)json.append(',');json.append("\n      {");appendArray(json,"attn_norm",model.layerWeight(b).values(),true,8);appendArray(json,"q",model.layerWeight(b+1).values(),true,8);appendArray(json,"k",model.layerWeight(b+2).values(),true,8);appendArray(json,"v",model.layerWeight(b+3).values(),true,8);appendArray(json,"o",model.layerWeight(b+4).values(),true,8);appendArray(json,"ffn_norm",model.layerWeight(b+5).values(),true,8);appendArray(json,"ffn_in",model.layerWeight(b+6).values(),true,8);appendArray(json,"ffn_out",model.layerWeight(b+7).values(),false,8);json.append("\n      }");}json.append("\n    ],");appendArray(json,"final_norm",model.finalNorm().values(),true);appendArray(json,"output",model.output().values(),false);json.append("\n  }\n}\n");Files.createDirectories(outputPath.toAbsolutePath().getParent());Files.writeString(outputPath,json,StandardCharsets.UTF_8);}

    private static double[] randomNormalArray(Random random, int size, double standardDeviation) {
        double[] values = new double[size];
        for (int index = 0; index < size; index++) values[index] = random.nextGaussian() * standardDeviation;
        return values;
    }
    private static double[] ones(int size) { double[] values = new double[size]; java.util.Arrays.fill(values, 1); return values; }
    private static void appendArray(StringBuilder json, String name, double[] values, boolean trailingComma) { appendArray(json, name, values, trailingComma, 4); }
    private static void appendArray(StringBuilder json, String name, double[] values, boolean trailingComma, int indent) {
        json.append("\n").append(" ".repeat(indent)).append('"').append(name).append("\": [");
        for (int index = 0; index < values.length; index++) { if (index > 0) json.append(','); json.append(number(values[index])); }
        json.append(']'); if (trailingComma) json.append(',');
    }
    private static String number(double value) { return String.format(Locale.ROOT, "%.9g", value); }
}
