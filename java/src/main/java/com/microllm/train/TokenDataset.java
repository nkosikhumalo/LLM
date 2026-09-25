package com.microllm.train;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.io.BufferedReader;
import java.io.PushbackReader;

/** Flat next-token dataset written by Go's tokenizer, with causal context windows. */
public final class TokenDataset {
    private final int[] tokens;

    private TokenDataset(int[] tokens) {
        if (tokens.length < 2) {
            throw new IllegalArgumentException("need at least two tokens");
        }
        this.tokens = tokens;
    }

    public static TokenDataset load(Path path) throws IOException {
        int[] values = new int[1024];
        int size = 0;
        try (PushbackReader reader = new PushbackReader(Files.newBufferedReader(path), 1)) {
            int current = skipWhitespace(reader);
            if (current != '[') throw new IOException("tokens file must be a JSON array of integer IDs");
            current = skipWhitespace(reader);
            if (current == ']') throw new IOException("tokens file is empty");
            while (true) {
                StringBuilder number = new StringBuilder();
                if (current == '-') {
                    number.append((char) current);
                    current = reader.read();
                }
                if (current < '0' || current > '9') throw new IOException("expected an integer token ID");
                do {
                    number.append((char) current);
                    current = reader.read();
                } while (current >= '0' && current <= '9');

                final int token;
                try {
                    token = Integer.parseInt(number.toString());
                } catch (NumberFormatException exception) {
                    throw new IOException("token ID is outside the supported integer range", exception);
                }
                if (token < 0) throw new IOException("token IDs cannot be negative");
                if (size == values.length) values = java.util.Arrays.copyOf(values, Math.multiplyExact(values.length, 2));
                values[size++] = token;

                while (current == ' ' || current == '\n' || current == '\r' || current == '\t') current = reader.read();
                if (current == ']') break;
                if (current != ',') throw new IOException("expected ',' or ']' after token ID");
                current = skipWhitespace(reader);
            }
            if (skipWhitespace(reader) != -1) throw new IOException("unexpected data after tokens array");
        }
        return new TokenDataset(java.util.Arrays.copyOf(values, size));
    }

    private static int skipWhitespace(PushbackReader reader) throws IOException {
        int current;
        do {
            current = reader.read();
        } while (current == ' ' || current == '\n' || current == '\r' || current == '\t');
        return current;
    }

    public int size() {
        return tokens.length;
    }

    /** Loads reply windows produced by the Go tokenizer. */
    public static List<AnswerAnchor> loadAnswerAnchors(Path path) throws IOException {
        List<AnswerAnchor> anchors = new ArrayList<>();
        try (var lines = Files.lines(path)) {
            lines.forEach(line -> {
                String[] fields = line.trim().split("\\t");
                if (fields.length != 2 && fields.length != 3) {
                    throw new IllegalArgumentException("expected window, answer, and optional end offsets");
                }
                try {
                    int windowStart = Integer.parseInt(fields[0]);
                    int answerStart = Integer.parseInt(fields[1]);
                    int targetLength = fields.length == 3
                            ? Integer.parseInt(fields[2]) - windowStart - 1
                            : 128;
                    int firstTargetIndex = answerStart - windowStart - 1;
                    if (windowStart < 0 || firstTargetIndex < 0 || targetLength <= firstTargetIndex) {
                        throw new IllegalArgumentException("answer anchor offsets are invalid");
                    }
                    anchors.add(new AnswerAnchor(windowStart, firstTargetIndex, targetLength));
                } catch (NumberFormatException exception) {
                    throw new IllegalArgumentException("invalid answer anchor: " + line, exception);
                }
            });
        } catch (IllegalArgumentException exception) {
            throw new IOException("invalid answer anchor file", exception);
        }
        return anchors;
    }

    public int inputAt(int index) {
        return tokens[index];
    }

    public int targetAt(int index) {
        return tokens[index + 1];
    }

    /** Last valid window start index (inclusive). */
    public int maxWindowOffset() {
        return tokens.length - 2;
    }

    /**
     * Splits the token stream into contiguous train/validation datasets.
     * Validation gets the final fraction of tokens when the corpus is large enough.
     */
    public Split split(double validationFraction) {
        if (validationFraction < 0.0 || validationFraction >= 1.0) {
            throw new IllegalArgumentException("validationFraction must be in [0, 1)");
        }
        if (validationFraction == 0.0 || tokens.length < 8) {
            return new Split(this, null);
        }
        int validationTokens = Math.max(2, (int) Math.round(tokens.length * validationFraction));
        int trainTokens = tokens.length - validationTokens;
        if (trainTokens < 2) {
            return new Split(this, null);
        }
        return new Split(slice(0, trainTokens), slice(trainTokens, tokens.length));
    }

    /** All valid window offsets for this dataset. */
    public List<Integer> windowOffsets() {
        return windowOffsets(1);
    }

    /** Returns valid window starts separated by the given token stride. */
    public List<Integer> windowOffsets(int stride) {
        if (stride < 1) throw new IllegalArgumentException("stride must be positive");
        List<Integer> offsets = new ArrayList<>((maxWindowOffset() / stride) + 1);
        for (int offset = 0; offset <= maxWindowOffset(); offset += stride) {
            offsets.add(offset);
        }
        return offsets;
    }

    /** Returns a bounded causal training window beginning at token offset. */
    public Window window(int offset, int maxLength) {
        if (maxLength < 1) {
            throw new IllegalArgumentException("maxLength must be positive");
        }
        if (offset < 0 || offset >= tokens.length - 1) {
            throw new IllegalArgumentException("invalid window offset");
        }
        int length = Math.min(maxLength, tokens.length - offset - 1);
        int[] input = new int[length];
        int[] target = new int[length];
        System.arraycopy(tokens, offset, input, 0, length);
        System.arraycopy(tokens, offset + 1, target, 0, length);
        return new Window(input, target);
    }

    private TokenDataset slice(int from, int to) {
        int[] copy = new int[to - from];
        System.arraycopy(tokens, from, copy, 0, copy.length);
        return new TokenDataset(copy);
    }

    public record Window(int[] input, int[] target) {}

    /** firstTargetIndex identifies the first answer token to supervise in a window. */
    public record AnswerAnchor(int windowStart, int firstTargetIndex, int targetLength) { }

    public record Split(TokenDataset train, TokenDataset validation) {}
}
