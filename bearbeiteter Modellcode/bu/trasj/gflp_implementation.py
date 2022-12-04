def _gather_final_log_probs(self,
                            generation_log_probs: torch.Tensor,
                            copy_log_probs: torch.Tensor,
                            state: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Combine copy probabilities with generation probabilities for matching tokens.

    Parameters
    ----------
    generation_log_probs : ``torch.Tensor``
        Shape: `(group_size, target_vocab_size)`
    copy_log_probs : ``torch.Tensor``
        Shape: `(group_size, trimmed_source_length)`
    state : ``Dict[str, torch.Tensor]``

    Returns
    -------
    torch.Tensor
        Shape: `(group_size, target_vocab_size + trimmed_source_length)`.
    """
    t1 = time.time()
    if (not self._reduced_copy_attention):
        _, trimmed_source_length = state["source_to_target"].size()
        source_token_ids = state["source_token_ids"]


        # shape: [(batch_size, *)]
        modified_log_probs_list: List[torch.Tensor] = []
        for i in range(trimmed_source_length):
            # shape: (group_size,)
            copy_log_probs_slice = copy_log_probs[:, i]
            # `source_to_target` is a matrix of shape (group_size, trimmed_source_length)
            # where element (i, j) is the vocab index of the target token that matches the jth
            # source token in the ith group, if there is one, or the index of the OOV symbol otherwise.
            # We'll use this to add copy scores to corresponding generation scores.
            # shape: (group_size,)
            source_to_target_slice = state["source_to_target"][:, i]
            # The OOV index in the source_to_target_slice indicates that the source
            # token is not in the target vocab, so we don't want to add that copy score
            # to the OOV token.
            copy_log_probs_to_add_mask = (source_to_target_slice != self._oov_index).float()
            copy_log_probs_to_add = copy_log_probs_slice + (copy_log_probs_to_add_mask + 1e-45).log()
            # shape: (batch_size, 1)
            copy_log_probs_to_add = copy_log_probs_to_add.unsqueeze(-1)
            # shape: (batch_size, 1)
            selected_generation_log_probs = generation_log_probs.gather(1, source_to_target_slice.unsqueeze(-1))
            combined_scores = util.logsumexp(
                    torch.cat((selected_generation_log_probs, copy_log_probs_to_add), dim=1))
            generation_log_probs = generation_log_probs.scatter(-1,
                                                                source_to_target_slice.unsqueeze(-1),
                                                                combined_scores.unsqueeze(-1))
            copy_log_probs_cpu = copy_log_probs.cpu()
            # We have to combine copy scores for duplicate source tokens so that
            # we can find the overall most likely source token. So, if this is the first
            # occurence of this particular source token, we add the log_probs from all other
            # occurences, otherwise we zero it out since it was already accounted for.
            if i < (trimmed_source_length - 1):
                # Sum copy scores from future occurences of source token.
                # shape: (group_size, trimmed_source_length - i)
                source_future_occurences = (source_token_ids[:, (i+1):] == source_token_ids[:, i].unsqueeze(-1)).float()  # pylint: disable=line-too-long
                # shape: (group_size, trimmed_source_length - i)
                future_copy_log_probs = copy_log_probs[:, (i+1):] + (source_future_occurences + 1e-45).log()
                # shape: (group_size, 1 + trimmed_source_length - i)
                combined = torch.cat((copy_log_probs_slice.unsqueeze(-1), future_copy_log_probs), dim=-1)
                # shape: (group_size,)  
                copy_log_probs_slice = util.logsumexp(combined)
            if i > 0:
                # Remove copy log_probs that we have already accounted for.
                # shape: (group_size, i)
                source_previous_occurences = source_token_ids[:, 0:i] == source_token_ids[:, i].unsqueeze(-1)
                # shape: (group_size,)
                duplicate_mask = (source_previous_occurences.sum(dim=-1) == 0).float()
                copy_log_probs_slice = copy_log_probs_slice + (duplicate_mask + 1e-45).log()

            # Finally, we zero-out copy scores that we added to the generation scores
            # above so that we don't double-count them.
            # shape: (group_size,)
            left_over_copy_log_probs = copy_log_probs_slice + (1.0 - copy_log_probs_to_add_mask + 1e-45).log()
            modified_log_probs_list.append(left_over_copy_log_probs.unsqueeze(-1))
        modified_log_probs_list.insert(0, generation_log_probs)

        # shape: (group_size, target_vocab_size + trimmed_source_length)
        modified_log_probs = torch.cat(modified_log_probs_list, dim=-1)


    elif self._reduced_copy_attention:
            #kombiniert nur nicht die Copy log probs
            # das Addieren der copy log probs der Vokabeln im Vokabelsatz zu den generative Log probs im ersten Teil 
            # von gather_final_log_probs war dagegen extrem wichtig. Beim Auslassen gab es Leistungseinbrüche. 
            _, trimmed_source_length = state["source_to_target"].size()
            source_token_ids = state["source_token_ids"]
            modified_log_probs_list: List[torch.Tensor] = []
            for i in range(trimmed_source_length):
                copy_log_probs_slice = copy_log_probs[:, i]
                source_to_target_slice = state["source_to_target"][:, i]
                copy_log_probs_to_add_mask = (source_to_target_slice != self._oov_index).float()
                copy_log_probs_to_add = copy_log_probs_slice + (copy_log_probs_to_add_mask + 1e-45).log()
                copy_log_probs_to_add = copy_log_probs_to_add.unsqueeze(-1)
                selected_generation_log_probs = generation_log_probs.gather(1, source_to_target_slice.unsqueeze(-1))
                combined_scores = util.logsumexp(
                        torch.cat((selected_generation_log_probs, copy_log_probs_to_add), dim=1))
                generation_log_probs = generation_log_probs.scatter(-1,
                                                                    source_to_target_slice.unsqueeze(-1),
                                                                    combined_scores.unsqueeze(-1))

            copy_log_probs_to_add_mask = (state["source_to_target"] != self._oov_index).float()                                                       
            left_over_copy_log_probs = copy_log_probs + (1.0 - copy_log_probs_to_add_mask + 1e-45).log()
            modified_log_probs = torch.cat([generation_log_probs, left_over_copy_log_probs], dim=-1)

    return modified_log_probs