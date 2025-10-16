
export COMET_API_KEY="x49hnrC7hy5M0xxzTUQBAlVrM"

python3 train.py -cn=conformer_librispeech_train.yaml \
 dataloader.batch_size=30 \
 trainer.override=False \
 writer=cometml \
 writer.run_name="conformer_30m"
