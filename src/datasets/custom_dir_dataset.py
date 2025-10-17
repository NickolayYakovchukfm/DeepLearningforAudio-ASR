from pathlib import Path

import torchaudio

from src.datasets.base_dataset import BaseDataset


class CustomDirDataset(BaseDataset):
    def __init__(self, data_dir, *args, **kwargs):
        audio_dir = Path(data_dir) / "audio"
        transcr_dir = Path(data_dir) / "transcriptions"

        index = []
        for audio_file in audio_dir.glob("*.*"):
            if audio_file.suffix in [".wav", ".flac", ".mp3"]:
                uid = audio_file.stem
                text_file = transcr_dir / f"{uid}.txt"
                text = text_file.read_text().strip() if text_file.exists() else ""

                info = torchaudio.info(str(audio_file))
                index.append(
                    {
                        "path": str(audio_file),
                        "text": text,
                        "audio_len": info.num_frames / info.sample_rate,
                        "utterance_id": uid,
                    }
                )
        super().__init__(index, *args, **kwargs)

    def __getitem__(self, ind):
        data = super().__getitem__(ind)
        data["utterance_id"] = self._index[ind]["utterance_id"]
        if "utterance_id" in self._index[ind]:
            data["utterance_id"] = self._index[ind]["utterance_id"]
        return data
