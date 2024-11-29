# Todo list of consultations

## Meeting 29.11.2024

Agreed with martin R. on creation of EKG dataset - generating relief maps for arterial fibrosis. Will be processed in step 2.

### Step 1
Learn how to use RRR on trivial dataset.
1. Create a toy MNIST probelm classifycing only between 0 and 8.
2. Train a CNN classifier on the data
3. generate attribution map of the classifier (should focus somewhere to the center)
4. Check if the attribution maps are correct if false, use RRR to fix them.

It is possible that the attribution maps will be perfect from the beginning. Then, it should be possible to generate artificial confounder, i.e. by adding a dot behing the zeros. 
GOTO 1.

### Step 2
Apply the previous approach to teh EKG dataset.
The attibution map of teh arterial fibrilation should be before the main peak and peak-to-peak variations. The `prohibited` areas should be the end of teh beat.
