import torch
def _gather_final_log_probs(self,
                            generation_log_probs: torch.Tensor,
                            copy_log_probs: torch.Tensor,
                            state: Dict[str, torch.Tensor]) -> torch.Tensor:
    _, trimmed_source_length = state["source_to_target"].size()
    source_token_ids = state["source_token_ids"]


    # shape: [(batch_size, *)]
    modified_log_probs_list: List[torch.Tensor] = []
    #for i in range(trimmed_source_length):
    #copy_log_probs_slice = copy_log_probs[:, i]
    #source_to_target_slice = state["source_to_target"][:, i]
    #copy_log_probs_to_add_mask = (source_to_target_slice != self._oov_index).float()
    copy_log_probs_to_add_mask = (state["source_to_target"] != self._oov_index).float()
    #copy_log_probs_to_add = copy_log_probs_slice + (copy_log_probs_to_add_mask + 1e-45).log()
    copy_log_probs_to_add = copy_log_probs + (copy_log_probs_to_add_mask + 1e-45).log()
    
    #copy_log_probs_to_add = copy_log_probs_to_add.unsqueeze(-1)
    copy_log_probs_to_add = copy_log_probs_to_add
    
    #selected_generation_log_probs = generation_log_probs.gather(1, source_to_target_slice.unsqueeze(-1))
    selected_generation_log_probs = generation_log_probs.gather(1, state["source_to_target"])
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

    return modified_log_probs