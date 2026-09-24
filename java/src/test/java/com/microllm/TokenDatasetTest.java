package com.microllm;
import com.microllm.train.TokenDataset;
import org.junit.jupiter.api.Test;
import java.nio.file.Files;
import static org.junit.jupiter.api.Assertions.*;
class TokenDatasetTest {
 @Test void windowsAreShiftedAndBounded() throws Exception { var file=Files.createTempFile("tokens",".json");Files.writeString(file,"[2, 4, 6, 8]");var data=TokenDataset.load(file);var w=data.window(1,8);assertArrayEquals(new int[]{4,6},w.input());assertArrayEquals(new int[]{6,8},w.target()); }
}
