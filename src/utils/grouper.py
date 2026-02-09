from src.utils.ppl_engine import PPLEngine
from config import config
import logging

logger = logging.getLogger(__name__)

def group_sequence_by_ppl_12(engine: PPLEngine, sequence: list) -> list:
    """
    Group a linear sequence of paragraph objects based on conditional PPL spikes.

    Args:
        engine (PPLEngine):
            Language model engine used to compute conditional perplexity.
        sequence (list):
            A list of paragraph dicts. Each element must contain at least:
            - 'text': paragraph text
            - 'depth': structural depth (used only for logging)

    Returns:
        list:
            A list of paragraph groups (List[List[Dict]]).
            Each inner list represents a contiguous semantic group.
    """
    if not sequence:
        return []

    if len(sequence) <= 2:
        return [sequence]

    START_SPLIT_RATIO = config['grouper']['START_SPLIT_RATIO']  # threshold to split the first paragraph

    groups = []

    ppl_0_1 = engine.compute_conditional_ppl(
        context=sequence[0]["text"],
        target=sequence[1]["text"],
    )

    ppl_1_2 = engine.compute_conditional_ppl(
        context=sequence[1]["text"],
        target=sequence[2]["text"],
    )

    logger.debug(
        "Init Check: PPL(0→1)=%.2f vs PPL(1→2)=%.2f",
        ppl_0_1,
        ppl_1_2,
    )

    if ppl_0_1 > ppl_1_2 * START_SPLIT_RATIO:
        logger.debug(
            "Start spike detected: %.2f > %.2f. Splitting first paragraph.",
            ppl_0_1,
            ppl_1_2,
        )
        groups.append([sequence[0]])
        current_group = [sequence[1]]
    else:
        current_group = [sequence[0], sequence[1]]

    prev_ppl = ppl_0_1

    for idx in range(2, len(sequence)):
        prev_para = sequence[idx - 1]
        curr_para = sequence[idx]

        if idx == 2:
            curr_ppl = ppl_1_2
        else:
            curr_ppl = engine.compute_conditional_ppl(
                context=prev_para["text"],
                target=curr_para["text"],
            )

        # Spike detection: PPL increases
        if prev_ppl < curr_ppl:
            logger.debug(
                "Spike detected at depth %s: %.2f → %.2f. Splitting group.",
                curr_para.get("depth"),
                prev_ppl,
                curr_ppl,
            )
            groups.append(current_group)
            current_group = [curr_para]
        else:
            current_group.append(curr_para)

        prev_ppl = curr_ppl

    if current_group:
        groups.append(current_group)

    return groups
