import re
from string import ascii_lowercase

import torch
from torchaudio.models import decoder

# TODO add CTC decode
# TODO add BPE, LM, Beam Search support
# Note: think about metrics and encoder
# The design can be remarkably improved
# to calculate stuff more efficiently and prettier


class CTCTextEncoder:
    EMPTY_TOK = ""

    def __init__(self, alphabet=None, lm_inf_path="4-gram.arpa", **kwargs):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """

        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")

        self.alphabet = alphabet
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)

        self.lm_inf_path = lm_inf_path

        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            return torch.Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
        except KeyError:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )

    def decode(self, inds) -> str:
        """
        Raw decoding without CTC.
        Used to validate the CTC decoding implementation.

        Args:
            inds (list): list of tokens.
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        return "".join([self.ind2char[int(ind)] for ind in inds]).strip()

    def ctc_beam_search(self, log_probs, log_probs_length, beam_size=80):
        log_probs = log_probs.cpu()
        log_probs_length = log_probs_length.cpu()

        beam_search = decoder.ctc_decoder(
            lexicon=None,
            tokens=self.vocab,
            lm=self.lm_inf_path,
            beam_size=beam_size,
            blank_token=self.EMPTY_TOK,
            sil_token=self.EMPTY_TOK,
        )

        beam_search_results = beam_search(log_probs, log_probs_length)
        res = []
        for beam_result in beam_search_results:
            res.append(beam_result[0].tokens)
        return res

    def ctc_decode(self, inds) -> str:
        """
        Decoding with CTC.
        """
        text = self.EMPTY_TOK
        prev_symb = self.EMPTY_TOK

        for index in inds:
            current = index

            if current == prev_symb:
                continue

            prev_symb = current
            text += self.ind2char[current]
        return text

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
