# IMoJIE-Faster-Copy-Attention

## Overview

This is a simple repository holding scrips that were used for a German 'Belegarbeit' about the [IMoJIE-Model](https://arxiv.org/abs/2005.08178).

If one searches to replicate the speed-up for IMoJIE from 247.4 to 85.8 seconds without a degradation in performance or any retraining, just copy code for the optimized gather_final_log_probs() function from this repo and may use it alongside the original [IMoJIE-Code](https://github.com/dair-iitd/imojie).

## Content

As this work targets teachers, it is written in a long but easier to understand manner for the general adept reader. Also, this work represents the writer's first ever paper that was written inside the field of AI and NLP. This belegarbeit started with the goal of replacing the BERT encoder of IMoJIE with more effective variants to potentially achieve a new state of the art performances. Firstly, however, experiments oriented on the easier task of replacing only the LSTM decoder with a GRU decoder for possible performance and speed improvements. No significant gains were made by replacing the decoder, but the huge performance bottle-neck of the gather_final_log_probs() function was found in the end and successfully optimized.

The second goal of replacing the BERT encoder with the more recent variants RoBERTa, ELECTRA, DeBERTa, DeBERTaV3 was done in [this repo](https://github.com/HenningBeyer/DetIE-with-DeBERTaV3). The deprecated version of AllenNLP alongside the BERT-oriented IMoJIE code made experiments very difficult, which is why experiments were finally conducted on the more recent DetIE model.

## Plots and Measurements

### Decoding requires over 95% of IMoJIE's the extraction time:
![Encoder-Decoder-Zeiten_für_github](https://user-images.githubusercontent.com/60894149/206861614-0e1ceecf-f47c-4fce-9ad0-85e194bc7442.png)

### Model performance wont deacrease by using the optimized function:
![optimized_extraction_speeds_for_github](https://user-images.githubusercontent.com/60894149/206860806-c9855525-3de7-4b8c-be3f-4b8b69fca294.png)
 Note that 'GRU' and 'LSTM' represent IMoJIE models with their respective decoder. Their performances are nearly similar. Numbers represent the training batch sizes for IMoJIE. <br>
 '*' means that no copy log probabilities of the exact same tokens are combined anymore. <br>
 '†' means that tokens will be only copied from the source sentence. This frames the combining/summing of token copy log probs just to the source sentences.

