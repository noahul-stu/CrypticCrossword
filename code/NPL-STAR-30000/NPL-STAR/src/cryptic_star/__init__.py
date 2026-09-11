"""Self-Taught Reasoner (STaR) generator for cryptic crosswords.

Stages, in the order the pipeline runs them:

    data.cryptonite  / data.wordplay   load the two datasets
    data.align                         join them -> human-rationale seed set
    data.formatting                    prompt + target templates
    train                              T5 fine-tuning
    generate                           sample N traces per clue
    verify                             label each trace 1 / 0
    rationalize                        second pass on failed clues, with a hint
    evaluate                           Exact Match + pass@N
    export_discriminator_data          hand-off to the DeBERTa re-ranker
    star_loop                          orchestrates all of the above
"""

__version__ = "0.1.0"
