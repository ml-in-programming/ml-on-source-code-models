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
        String array[] = {"Ron", "Harry", "Hermoine"};

        //enhanced for loop
        for (String x:array)
        {
            System.out.println(x);
        }

        int x = 1;

        // Exit when x becomes greater than 4
        while (x <= 4)
        {
            System.out.println("Value of x:" + x);

            // Increment the value of x for
            // next iteration
            x++;
        }

        x = 21;
        do
        {
            // The line will be printed even
            // if the condition is false
            System.out.println("Value of x:" + x);
            x++;
        }
        while (x < 20);

        int с = 0;
        this.runningBit = (a & 1) != 0;
        int dD = 20;
        int notLowerCaseNotInitialized; }
}