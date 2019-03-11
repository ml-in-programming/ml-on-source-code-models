package com.googlecode.javaewah;

public final class BufferedRunningLengthWord
  {
    // Running bit
    protected boolean runningBit; // line comment

    /**
     * Instantiates a new buffered running length word.
     *
     * @param a the word
     */
    public BufferedRunningLengthWord(final long a) {
        this.runningBit = (a & 1) != 0;  }
}