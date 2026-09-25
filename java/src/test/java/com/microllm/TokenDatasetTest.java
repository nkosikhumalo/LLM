package com.microllm;

import com.microllm.train.TokenDataset;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;

class TokenDatasetTest {
    @Test
    void windowsAreShiftedAndBounded() throws Exception {
        var file = Files.createTempFile("tokens", ".json");
        Files.writeString(file, "[2, 4, 6, 8]");
        var data = TokenDataset.load(file);
        var window = data.window(1, 8);
        assertArrayEquals(new int[]{4, 6}, window.input());
        assertArrayEquals(new int[]{6, 8}, window.target());
    }

    @Test
    void answerAnchorRetainsTheReplyEndBoundary() throws Exception {
        var file = Files.createTempFile("answer-anchors", ".tsv");
        Files.writeString(file, "4\t8\t13\n");

        var anchors = TokenDataset.loadAnswerAnchors(file);

        assertEquals(1, anchors.size());
        assertEquals(new TokenDataset.AnswerAnchor(4, 3, 8), anchors.get(0));
    }

    @Test
    void splitCreatesValidationTail() throws Exception {
        var file = Files.createTempFile("tokens", ".json");
        Files.writeString(file, "[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]");
        TokenDataset.Split split = TokenDataset.load(file).split(0.2);
        assertEquals(8, split.train().size());
        assertNotNull(split.validation());
        assertEquals(2, split.validation().size());
    }

    @Test
    void tinyCorpusSkipsValidation() throws Exception {
        var file = Files.createTempFile("tokens", ".json");
        Files.writeString(file, "[1, 2, 3]");
        TokenDataset.Split split = TokenDataset.load(file).split(0.1);
        assertNull(split.validation());
        assertEquals(3, split.train().size());
    }
}
