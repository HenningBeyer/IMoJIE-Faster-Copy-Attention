# IMoJIE-Faster-Copy-Attention

## Overview

This is a simple repository holding scrips that were used for a German 'Besondere Lernleistung' about the [IMoJIE-Model](https://arxiv.org/abs/2005.08178) alongside the self-written BeLL paper.

If one searches to replicate the speed-up for IMoJIE from 247.4 to 85.8 seconds without a degradation in performance or any retraining, just copy code for the optimized gather_final_log_probs() function from this repo and may use it alongside the original [IMoJIE-Code](https://github.com/dair-iitd/imojie).

## Content

As this work targets teachers, it is written in a long but easier to understand manner suitable for any adept reader. It started with the goal to replace the BERT encoder of IMoJIE with a more effective variant to potentially achieve a new state of the art performance, however, firstly the simpler task of replacing only the LSTM decoder with a GRU decoder was completed. In summary, no significant gains were made by replacing the decoder, but a huge performance bottle-neck of the gather_final_log_probs() function was found and successfully optimized.

The second goal considered the replacing of the BERT encoder with more recent variants such as RoBERTa, ELECTRA, DeBERTa, DeBERTaV3. Here the very huge code of IMoJIE presented a big problem and caused the switching to DetIE for completing these experiments. For that and a more compact paper only regarding the second goal, refer to [this repo](https://github.com/HenningBeyer/DetIE-with-DeBERTaV3).


## Plots and Measurements

### Decoding requires over 95% of the extraction time:
![Encoder-Decoder-Zeiten_für_github](https://user-images.githubusercontent.com/60894149/206861614-0e1ceecf-f47c-4fce-9ad0-85e194bc7442.png)

### Model performance wont deacrease by using the optimized function:
![optimized_extraction_speeds_for_github](https://user-images.githubusercontent.com/60894149/206860806-c9855525-3de7-4b8c-be3f-4b8b69fca294.png)
 Note that 'GRU' and 'LSTM' represent IMoJIE models with their respective decoder. Numbers represent the training batch size. <br>
 '*' means that no copy log probabilities of similar tokens are combined anymore. <br>
 '†' means that tokens will only be copied from the source sentence. This frames the combining of token copy log probs to the source sentence.

