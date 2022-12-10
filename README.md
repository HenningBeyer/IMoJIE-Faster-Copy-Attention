# IMoJIE_Faster_Copy_Attention

This is a simple repository holding scrips that were used for a German 'Besondere Lernleistung' alongside the provided paper.

If one searches to replicate the speed-up for IMoJIE from 247.4 to 85.8 seconds without a degradation in performance or any retraining, just copy code for the optimized gather_final_log_probs() function from this repo.

As this work targets teachers it is written in a long but easier to understand manner suitable for any adept reader. It started with the goal to replace the BERT encoder of IMoJIE to potentially achieve a new state of the art performance, however, firstly the simpler task of replacing the LSTM decoder with a GRU decoder more suitable for a student. In summary, no significant gains were made by replacing the decoder, but a huge performance bottle-neck of the gather_final_log_probs() function was found and  successfully optimized.

The second goal considered the replacing of the BERT encoder with more recent variants such as RoBERTa, ELECTRA, DeBERTa, DeBERTaV3. Here the very huge code of IMoJIE presented a big problem and caused the switching to DetIE for completing these experiments. For that and a more compact paper only regarding the second goal, refer to [this repo](https://github.com/HenningBeyer/DetIE-with-DeBERTaV3).
 

