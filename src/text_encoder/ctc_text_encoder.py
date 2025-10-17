import re
from collections import defaultdict
from string import ascii_lowercase

import torch
from pyctcdecode import build_ctcdecoder
from torchaudio.models import decoder

from download_lm import fetch_conformer_model_weights, fetch_language_model


class CTCTextEncoder:
    EMPTY_TOK = ""

    def __init__(self, alphabet=None, lm_inf_path="4-gram.arpa", **kwargs):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """
        # fetch_conformer_model_weights()
        # fetch_language_model()
        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")

        self.alphabet = alphabet
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)

        self.lm_inf_path = lm_inf_path
        if self.lm_inf_path is not None:
            self.decoder = build_ctcdecoder(
                self.vocab,
                kenlm_model_path=self.lm_inf_path,
            )
        else:
            self.decoder = build_ctcdecoder(self.vocab, kenlm_model_path=None)

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

    def ctc_beam_search(self, log_probs, beam_size=100):
        if isinstance(log_probs, torch.Tensor):
            log_probs = log_probs.detach().cpu().numpy()
        result = self.decoder.decode(log_probs, beam_size)
        return result

    def ctc_beam_search_lm(self, log_probs, beam_size=100):
        if isinstance(log_probs, torch.Tensor):
            log_probs = log_probs.detach().cpu().numpy()
        return self.decoder.decode(log_probs, beam_size)

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

    def expand_and_merge_beams(self, dp, cur_step_prob, ind2char):
        new_dp = defaultdict(float)

        for (pref, prev_char), pref_proba in dp.items():
            for idx, char in ind2char.items():
                cur_proba = pref_proba * cur_step_prob[idx]
                cur_char = char

                if char == self.EMPTY_TOK:
                    cur_pref = pref
                else:
                    if prev_char != char:
                        cur_pref = pref + char
                    else:
                        cur_pref = pref

                new_dp[(cur_pref, cur_char)] += cur_proba
        return new_dp

    def truncate_beams(self, dp, beam_size):
        return dict(
            sorted(list(dp.items()), key=lambda x: x[1], reverse=True)[:beam_size]
        )

    def ctc_beam_search_from_scratch(self, probs, beam_size):
        dp = {
            ("", self.EMPTY_TOK): 1.0,
        }
        for cur_step_prob in probs:
            dp = self.expand_and_merge_beams(dp, cur_step_prob, self.ind2char)
            dp = self.truncate_beams(dp, beam_size)

        result = [
            {"pref": pref} for (pref, _), _ in sorted(dp.items(), key=lambda x: -x[1])
        ][0]
        return result["pref"]

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
