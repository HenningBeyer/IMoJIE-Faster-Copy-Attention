# IMoJIE_Faster_Copy_Attention

This is a simple repository holding scrips that were used for a German 'Besondere Lernleistung' alongside the paper.

If one searches to replicate the speed-up from 247.4 to 85.8 Seconds for IMoJIE without a degradation in performance or any retraining, just copy code for the optimized gather_final_log_probs() function from the repo.

As this work targets teachers it is written in a long but easier to understand manner suitable for any adept reader. It may also contain descriptions for solving the difficult task as a single student with some fallbacks. In summary, the first goal was to achive any speed-up or performance increase by replacing IMoJIE's LSTM decoder with a GRU decoder where both hypotheses turned out to be wrong. However, they have hinted the huge performance bottle-neck of the gather_final_log_probs() function which could successfully be optimized.

A second goal considered the replacing of the BERT encoder with more recent variants such as RoBERTa, ELECTRA, DeBERTa, DeBERTaV3. Here the very huge code of IMoJIE presented a big problem and caused the switching to DetIE for completing these experiments. For that refer to (this repo)[https://github.com/HenningBeyer/DetIE-with-DeBERTaV3].
 

